Local RAG Document Assistant with Semantic Chunking & Automated Evaluation
An advanced, privacy-focused Retrieval-Augmented Generation (RAG) system that allows users to upload multiple PDF files and perform semantic Q&A locally. This application goes beyond standard RAG tutorials by implementing Semantic Document Chunking, a Two-Stage Retrieval Architecture, and a Real-Time Mathematical Evaluation Suite to guarantee answer fidelity and eliminate hallucinations.

Key Architectural Features
Dynamic Semantic Chunking: Replaces naive character-based splitting with LangChain's SemanticChunker. It utilizes local HuggingFace embeddings to "read" the document and group sentences by mathematical meaning, dynamically creating chunks based on contextual breakpoints rather than arbitrary word counts.

Two-Stage Intelligent Retrieval: * Stage 1 (Recall): Queries a local ChromaDB vector store to retrieve a wide net of top-10 contextually relevant document chunks.

Stage 2 (Precision): Applies Cohere Rerank (rerank-english-v3.0) via a ContextualCompressionRetriever to score, deeply evaluate, and filter down the highest quality chunks before presenting them to the LLM.

Automated Response Evaluation (RAG Metrics): Features a custom, zero-shot evaluation pipeline using cosine similarity math (via NumPy/Scikit-learn) to score the LLM's output in real-time across four critical enterprise metrics:

Faithfulness: Verifies the answer is mathematically grounded in the retrieved chunks.

Answer Relevance: Measures semantic alignment between the user's prompt and the final answer.

Context Recall: Evaluates the quality of ChromaDB's retrieval capabilities.

Coherence: A rule-based heuristic engine to penalize repetition, formatting errors, or out-of-scope refusals.

Local Execution: Uses OllamaLLM to interface directly with a local llama3.1 instance, keeping data safe and offline.

Transparent UI & Audit Trail: The Streamlit interface maintains a persistent memory of all uploaded documents in the database. Every AI response includes a drop-down audit panel displaying the exact source chunks used and the calculated RAG Evaluation scores with a visual progress bar system.

Tech Stack
Frontend UI: Streamlit

Orchestration: LangChain (langchain-core, langchain-community, langchain-experimental)

Evaluation Math: NumPy, Scikit-learn

Vector Database: Chroma (Persistent Local Instance)

Text Processing: SemanticChunker & PyPDFLoader

Embeddings: all-MiniLM-L6-v2 via HuggingFace

Reranking Engine: Cohere Rerank API

Local LLM Host: Ollama (Llama 3.1)
