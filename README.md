# Local RAG Document Assistant with Semantic Chunking & Automated Evaluation

An advanced, privacy-focused Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files and perform semantic Q&A locally. This application utilizes semantic document chunking, an advanced two-stage retrieval process, and a real-time mathematical evaluation suite to maximize retrieval accuracy and eliminate hallucinations.

##  Key Architectural Features
* **Dynamic Semantic Chunking:** Replaces naive character splitting with `SemanticChunker`, utilizing HuggingFace embeddings to group sentences by mathematical meaning and contextual breakpoints.
* **Two-Stage Intelligent Retrieval:** * *Stage 1 (Recall):* Queries a local **Chroma DB** vector store to retrieve a wide net of top-10 contextually relevant document chunks.
    * *Stage 2 (Precision):* Applies **Cohere Rerank (`rerank-english-v3.0`)** via a `ContextualCompressionRetriever` to score, evaluate, and filter down the highest quality chunks before presenting them to the LLM.
* **Automated Response Evaluation:** * *Faithfulness & Relevance:* Uses cosine similarity math to verify that answers are mathematically grounded in the retrieved context and accurately address the prompt.
    * *Context Recall & Coherence:* Evaluates Chroma DB's retrieval quality and utilizes a rule-based engine to penalize formatting errors or poor response structures.
* **Local Execution:** Uses `OllamaLLM` to interface directly with a local `llama3.1` instance, keeping data safe and offline.
* **Source Citations & Audit Trail:** Interactive UI includes full expanders detailing the exact document chunks used by the agent, alongside real-time visual progress bars for the RAG Evaluation scores.
* **Strict Guardrails:** Programmed system instructions eliminate hallucinations by forcing the model to decline answering if the context does not contain the solution.

##  Tech Stack
* **Frontend UI:** Streamlit
* **Orchestration:** LangChain (`langchain-core`, `langchain-community`, `langchain-experimental`)
* **Evaluation Math:** NumPy & Scikit-learn
* **Vector Database:** Chroma (Persistent Local Instance)
* **Text Processing:** SemanticChunker & PyPDFLoader
* **Embeddings:** `all-MiniLM-L6-v2` via HuggingFace
* **Reranking Engine:** Cohere Rerank API
* **Local LLM Host:** Ollama (Llama 3.1)
