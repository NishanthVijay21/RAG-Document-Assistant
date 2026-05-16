# Local RAG Document Assistant with Two-Stage Retrieval (Cohere Rerank)

An advanced, privacy-focused Retrieval-Augmented Generation (RAG) system that allows users to upload PDF files and perform semantic Q&A locally. This application utilizes an advanced two-stage retrieval process to maximize retrieval accuracy while keeping context windows efficient.

##  Key Architectural Features
*   **Two-Stage Intelligent Retrieval:** 
    *   *Stage 1 (Recall):* Queries a local **Chroma DB** vector store to retrieve a wide net of top-10 contextually relevant document chunks.
    *   *Stage 2 (Precision):* Applies **Cohere Rerank (`rerank-english-v3.0`)** via a `ContextualCompressionRetriever` to score, evaluate, and filter down the highest quality chunks before presenting them to the LLM.
*   **Local Execution:** Uses `OllamaLLM` to interface directly with a local `llama3` instance, keeping data safe and offline.
*   **Source Citations:** Interactive UI includes full expanders detailing the exact document chunks used by the agent to construct its answers, including rerank relevancy mapping.
*   **Strict Guardrails:** Programmed system instructions eliminate hallucinations by forcing the model to decline answering if the context does not contain the solution.

##  Tech Stack
*   **Frontend UI:** Streamlit
*   **Orchestration:** LangChain (`langchain-core`, `langchain-community`)
*   **Vector Database:** Chroma (Persistent Local Instance)
*   **Text Processing:** RecursiveCharacterTextSplitter & PyPDFLoader
*   **Embeddings:** `all-MiniLM-L6-v2` via HuggingFace
*   **Reranking Engine:** Cohere Rerank API
*   **Local LLM Host:** Ollama (Llama 3)
