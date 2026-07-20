import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_cohere import CohereRerank
from langchain_classic.retrievers import ContextualCompressionRetriever
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import tempfile
load_dotenv()
st.set_page_config(page_title="Local RAG chatbot", layout="wide")
st.title("🤖 Gemini Document Assistant")
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []
# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def get_embeddings_model():
    """Reuse the same embedding model for evaluation."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def embed_texts(texts: list[str], embedder) -> np.ndarray:
    return np.array(embedder.embed_documents(texts))


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    return float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])


# ── Metric 1: Faithfulness ───────────────────
def evaluate_faithfulness(answer: str, source_docs: list) -> dict:
    """
    Checks whether every sentence in the answer is semantically supported
    by at least one retrieved chunk.

    Score = fraction of answer sentences with max cosine-sim ≥ threshold.
    """
    THRESHOLD = 0.40 

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10]
    if not sentences:
        return {"score": 1.0, "details": "No scorable sentences found.", "supported": [], "unsupported": []}

    context_texts = [doc.page_content for doc in source_docs]
    if not context_texts:
        return {"score": 0.0, "details": "No context documents available.", "supported": [], "unsupported": sentences}

    embedder = get_embeddings_model()
    sentence_embs = embed_texts(sentences, embedder)
    context_embs  = embed_texts(context_texts, embedder)

    supported, unsupported = [], []
    for sent, s_emb in zip(sentences, sentence_embs):
        sims = [cosine_sim(s_emb, c_emb) for c_emb in context_embs]
        if max(sims) >= THRESHOLD:
            supported.append(sent)
        else:
            unsupported.append(sent)

    score = len(supported) / len(sentences)
    detail = f"{len(supported)}/{len(sentences)} answer sentences are grounded in the retrieved context."
    return {"score": score, "details": detail, "supported": supported, "unsupported": unsupported}


# ── Metric 2: Answer Relevance ───────────────
def evaluate_answer_relevance(question: str, answer: str) -> dict:
    """
    Measures semantic similarity between the question and the answer.
    High score → the answer actually addresses what was asked.
    """
    if not answer.strip():
        return {"score": 0.0, "details": "Empty answer."}

    embedder = get_embeddings_model()
    q_emb = embed_texts([question], embedder)[0]
    a_emb = embed_texts([answer],   embedder)[0]
    score = cosine_sim(q_emb, a_emb)

    if score >= 0.75:
        label = "High — answer closely addresses the question."
    elif score >= 0.50:
        label = "Moderate — answer is somewhat related to the question."
    else:
        label = "Low — answer may be off-topic."

    return {"score": score, "details": label}


# ── Metric 3: Context Recall (Proxy) ─────────
def evaluate_context_recall(question: str, source_docs: list) -> dict:
    """
    Measures how semantically relevant the retrieved chunks are to the question.
    Acts as a proxy for retrieval quality (true recall needs gold answers).
    Average cosine-sim between question and each retrieved chunk.
    """
    if not source_docs:
        return {"score": 0.0, "details": "No context retrieved.", "chunk_scores": []}

    embedder = get_embeddings_model()
    q_emb = embed_texts([question], embedder)[0]
    chunk_texts = [doc.page_content for doc in source_docs]
    chunk_embs  = embed_texts(chunk_texts, embedder)

    chunk_scores = [cosine_sim(q_emb, c_emb) for c_emb in chunk_embs]
    avg = float(np.mean(chunk_scores))

    label = "Good retrieval — chunks are relevant to the question." if avg >= 0.45 \
            else "Weak retrieval — chunks may not fully cover the question."

    return {"score": avg, "details": label, "chunk_scores": chunk_scores}


# ── Metric 4: Answer Coherence ───────────────
def evaluate_coherence(answer: str) -> dict:
    """
    Lightweight rule-based coherence check:
    - Length check (too short / too long)
    - Repetition penalty
    - Refusal detection
    """
    word_count   = len(answer.split())
    sentences    = re.split(r"(?<=[.!?])\s+", answer)
    unique_ratio = len(set(answer.lower().split())) / max(word_count, 1)
    is_refusal   = "not available in the uploaded document" in answer.lower()

    issues = []
    if word_count < 10:
        issues.append("Answer is very short.")
    if unique_ratio < 0.40:
        issues.append("High word repetition detected.")
    if is_refusal:
        issues.append("Model returned a refusal / out-of-scope response.")

    score = max(0.0, 1.0 - 0.25 * len(issues))
    detail = " | ".join(issues) if issues else "Answer appears coherent."
    return {"score": score, "details": detail,
            "word_count": word_count, "unique_ratio": round(unique_ratio, 2)}


# ── Aggregate Runner ─────────────────────────
def run_evaluation(question: str, answer: str, source_docs: list) -> dict:
    """Run all four metrics and return a combined report."""
    faithfulness    = evaluate_faithfulness(answer, source_docs)
    answer_relevance = evaluate_answer_relevance(question, answer)
    context_recall  = evaluate_context_recall(question, source_docs)
    coherence       = evaluate_coherence(answer)

    overall = np.mean([
        faithfulness["score"],
        answer_relevance["score"],
        context_recall["score"],
        coherence["score"],
    ])

    return {
        "overall":          round(overall, 3),
        "faithfulness":     faithfulness,
        "answer_relevance": answer_relevance,
        "context_recall":   context_recall,
        "coherence":        coherence,
    }


# ─────────────────────────────────────────────
# EVALUATION UI RENDERER
# ─────────────────────────────────────────────

def render_score_bar(label: str, score: float, details: str):
    """Render a labeled progress bar with colour coding."""
    colour = "#2ecc71" if score >= 0.70 else "#f39c12" if score >= 0.45 else "#e74c3c"
    pct = int(score * 100)
    st.markdown(f"""
    <div style="margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
            <span><b>{label}</b></span>
            <span style="color:{colour};font-weight:700;">{pct}%</span>
        </div>
        <div style="background:#2b2b2b;border-radius:6px;height:8px;margin:4px 0;">
            <div style="width:{pct}%;background:{colour};height:8px;border-radius:6px;
                        transition:width 0.6s ease;"></div>
        </div>
        <div style="font-size:0.75rem;color:#aaa;">{details}</div>
    </div>
    """, unsafe_allow_html=True)


def render_evaluation_panel(eval_results: dict):
    """Render the full evaluation panel inside an expander."""
    overall_pct = int(eval_results["overall"] * 100)
    colour = "#2ecc71" if overall_pct >= 70 else "#f39c12" if overall_pct >= 45 else "#e74c3c"

    with st.expander("📊 Response Evaluation Metrics", expanded=False):
        st.markdown(f"""
        <div style="text-align:center;padding:12px 0 18px;">
            <div style="font-size:0.8rem;color:#aaa;letter-spacing:1px;text-transform:uppercase;">
                Overall RAG Score
            </div>
            <div style="font-size:3rem;font-weight:800;color:{colour};line-height:1.1;">
                {overall_pct}%
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        render_score_bar(
            "🔗 Faithfulness",
            eval_results["faithfulness"]["score"],
            eval_results["faithfulness"]["details"],
        )
        render_score_bar(
            "🎯 Answer Relevance",
            eval_results["answer_relevance"]["score"],
            eval_results["answer_relevance"]["details"],
        )
        render_score_bar(
            "📚 Context Recall",
            eval_results["context_recall"]["score"],
            eval_results["context_recall"]["details"],
        )
        render_score_bar(
            "💬 Coherence",
            eval_results["coherence"]["score"],
            eval_results["coherence"]["details"],
        )

        coh = eval_results["coherence"]
        st.caption(
            f"Word count: {coh['word_count']} | "
            f"Vocabulary diversity: {int(coh['unique_ratio']*100)}%"
        )

        faith = eval_results["faithfulness"]
        if faith.get("unsupported"):
            with st.expander("⚠️ Potentially ungrounded sentences", expanded=False):
                for s in faith["unsupported"]:
                    st.warning(s)

        cr = eval_results["context_recall"]
        if cr.get("chunk_scores"):
            with st.expander("🔬 Per-chunk relevance scores", expanded=False):
                for i, sc in enumerate(cr["chunk_scores"]):
                    bar_w = int(sc * 100)
                    colour_c = "#2ecc71" if sc >= 0.45 else "#e74c3c"
                    st.markdown(f"""
                    <div style="margin-bottom:6px;">
                        <span style="font-size:0.8rem;">Chunk {i+1}</span>
                        <div style="background:#2b2b2b;border-radius:4px;height:6px;margin:2px 0;">
                            <div style="width:{bar_w}%;background:{colour_c};height:6px;border-radius:4px;"></div>
                        </div>
                        <span style="font-size:0.75rem;color:#aaa;">{round(sc,3)}</span>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 1. INDEXING LOGIC
# ─────────────────────────────────────────────

def process_document(uploaded_file):
    # Save PDF temporarily to extract text, then delete the temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    data = loader.load()
    os.remove(tmp_path)  # Clean up temp file immediately

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    chunks = text_splitter.split_documents(data)

    # Store in-memory inside THIS user's session state (No persist_directory)
    if st.session_state.vector_db is None:
        st.session_state.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
    else:
        st.session_state.vector_db.add_documents(chunks)

    if uploaded_file.name not in st.session_state.uploaded_docs:
        st.session_state.uploaded_docs.append(uploaded_file.name)

    return len(chunks)

# ─────────────────────────────────────────────
# 2. SIDEBAR UI
# ─────────────────────────────────────────────

with st.sidebar:
    st.header("Upload documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
            if st.button("Process & Index Document"):
                with st.spinner(f"Analysing {uploaded_file.name}…"):
                    num_chunks = process_document(uploaded_file)
                    st.success(f"Indexed {num_chunks} chunks! You can now ask questions.")
    st.divider()
    st.header("📚 Session Documents")
    if st.session_state.uploaded_docs:
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"- 📄 `{doc}`")
            
        # Optional: Add a Reset button to clear session memory
        if st.button("🗑️ Clear My Documents"):
            st.session_state.vector_db = None
            st.session_state.uploaded_docs = []
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("No documents uploaded in this session.")

    st.divider()
    st.markdown("**Evaluation legend**")
    st.markdown("🟢 ≥ 70%  · Good")
    st.markdown("🟡 45–69% · Moderate")
    st.markdown("🔴 < 45%  · Needs review")


# ─────────────────────────────────────────────
# 3. RAG SYSTEM SETUP
# ─────────────────────────────────────────────

def load_rag_system():
    if st.session_state.vector_db is None:
            return None

    vector_db = st.session_state.vector_db
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    compressor = CohereRerank(
        model="rerank-english-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

    system_prompt = (
        "You are a strict document search assistant. "
        "Use ONLY the provided context to answer the question. "
        "If the answer is not contained within the context, exactly say: "
        "'I am sorry, but that information is not available in the uploaded document.' "
        "\n\n"
        "{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, prompt_template)
    return create_retrieval_chain(compression_retriever, document_chain)


# ─────────────────────────────────────────────
# 4. CHAT INTERFACE
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("🔍 View Source Chunks"):
                    for i, doc in enumerate(message["sources"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.info(doc.page_content)
                        st.markdown("---")

            if message.get("eval_results"):
                render_evaluation_panel(message["eval_results"])


if prompt := st.chat_input("Ask something about your document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        qa = load_rag_system()
        if qa:
            with st.spinner("Searching database and thinking…"):
                response    = qa.invoke({"input": prompt})
                answer      = response["answer"]
                source_docs = response["context"]

            st.markdown(answer)

            with st.expander("🔍 View Source Chunks"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.info(doc.page_content)
                    st.markdown("---")

            with st.spinner("Evaluating response quality…"):
                eval_results = run_evaluation(prompt, answer, source_docs)

            render_evaluation_panel(eval_results)

            st.session_state.messages.append({
                "role":         "assistant",
                "content":      answer,
                "sources":      source_docs,
                "eval_results": eval_results,
            })
        else:
            st.error("Please upload and 'Process' a document in the sidebar first!")
