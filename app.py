import streamlit as st
from langchain_ollama import OllamaLLM
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

st.set_page_config(page_title="Local RAG chatbot", layout="wide")
st.title("🤖 Llama 3 Document Assistant")

# --- 1. INDEXING LOGIC ---
def process_document(file_path):
    # Load and split the PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Initialize Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create Vector Database (This overwrites/updates the local folder)
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    return len(chunks)

# --- 2. SIDEBAR UI ---
with st.sidebar:
    st.header("Upload documents")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # THE MISSING BUTTON
        if st.button("Process & Index Document"):
            with st.spinner("Analyzing PDF..."):
                num_chunks = process_document("temp.pdf")
                st.success(f"Indexed {num_chunks} chunks! You can now ask questions.")
                # Clear the cache to ensure the new DB is loaded
                st.cache_resource.clear()

# --- 3. RAG SYSTEM SETUP ---
@st.cache_resource
def load_rag_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists("./chroma_db"):
        vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    else:
        return None

    llm = OllamaLLM(model="llama3", temperature=0)

    # --- RERANKER LOGIC ---
    # 1. First, grab a wider net of documents (Stage 1: Recall)
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # 2. Define the Reranker (Stage 2: Precision)
    # Get a free trial key at dashboard.cohere.com
    compressor = CohereRerank(
        model="rerank-english-v3.0", 
        cohere_api_key="COHERE_API_KEY" 
    )

    # 3. Create the Compression Retriever
    # This wraps the base retriever and applies the reranking math
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
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
    
    # Use the smart compression_retriever instead of the basic one
    return create_retrieval_chain(compression_retriever, document_chain)

# --- 4. CHAT INTERFACE ---
# --- 4. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (now with sources!)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message has source documents, display them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Source Chunks"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1} (Score/Rank based on Cohere):**")
                    st.info(doc.page_content) # st.info puts it in a nice highlighted box
                    st.markdown("---")

if prompt := st.chat_input("Ask something about your document"):
    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        qa = load_rag_system()
        if qa:
            with st.spinner("Searching database and thinking..."):
                # Run the chain
                response = qa.invoke({"input": prompt})
                answer = response["answer"]
                source_docs = response["context"] # <--- HERE IS WHERE WE GET THE CHUNKS
                
                # Display the main answer
                st.markdown(answer)
                
                # Display the source chunks in a dropdown
                with st.expander("🔍 View Source Chunks"):
                    for i, doc in enumerate(source_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.info(doc.page_content)
                        st.markdown("---")
                
                # Save assistant message WITH the sources attached
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_docs # Saving it so it stays in history
                })
        else:
            st.error("Please upload and 'Process' a document in the sidebar first!")
