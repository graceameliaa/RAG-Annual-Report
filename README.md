# RAG Chatbot for Complex Financial Reports
This project is an advanced Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about a 588-page, real-world corporate annual report.

Unlike simple RAG tutorials that use plain text, this project is built to handle the "messy" reality of complex documents, including embedded tables, multi-column layouts, and semantic headings.

## The Problem
Standard RAG pipelines that use basic PDF text extractors fail on complex documents. They destroy table data, mix up text from different columns, and lose the document's structure, leading to inaccurate answers or hallucinations.

This project solves that by implementing a layout-aware ingestion and a retrieve-rerank pipeline.

## RAG Pipeline
This diagram shows the two-stage process I built: a one-time "Ingestion" pipeline and a real-time "Retrieval" pipeline.


```mermaid
graph TD
    subgraph "Ingestion Pipeline (Offline)"
        direction TB
        A[PDF Report (588 pages)] --> B(pymupdf4llm);
        B --> C[Markdown Text];
        C --> D(MarkdownTextSplitter);
        D --> E[Semantic Chunks (w/ Tables)];
        E --> F(HuggingFace Embeddings);
        F --> G[Vectors];
        G --> H[(ChromaDB)];
    end

    subgraph "Retrieval Pipeline (Online)"
        direction TB
        I[User Query] --> J(HuggingFace Embeddings);
        J --> K[Query Vector];
        K --> L[(ChromaDB)];
        L -- "1. Retrieve Top 25 Candidates" --> M[25 Chunks];
        
        M --> N(CrossEncoder Reranker);
        I --> N;
        N -- "2. Rerank for Relevance" --> O[Top 5 Chunks];
        
        O --> P{Prompt Template};
        I --> P;
        P -- "Final Context + Query" --> Q(Ollama - Mistral);
        Q --> R[Final Answer];
    end
```
