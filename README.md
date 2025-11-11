# 🤖 RAG Chatbot for Complex Financial Reports 📊
This project is an advanced Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about a 588-page, real-world corporate annual report.

Unlike simple RAG tutorials that use plain text, this project is built to handle the "messy" reality of complex documents, including embedded tables, multi-column layouts, and semantic headings.

-----

## ⚠️ The Problem
Standard RAG pipelines that use basic PDF text extractors fail on complex documents. They destroy table data, mix up text from different columns, and lose the document's structure, leading to inaccurate answers or hallucinations.

This project solves that by implementing a layout-aware ingestion and a retrieve-rerank pipeline.

-----

## 🔑 Key Features

1.  **Layout-Aware PDF Parsing:** Uses `pymupdf4llm` to convert the entire 588-page PDF into clean Markdown. This preserves all tables, lists, and headers, allowing the LLM to read financial data correctly.

2.  **Structure-Aware Chunking:** Employs a `MarkdownTextSplitter` to create semantic chunks based on the document's structure, keeping headers and their associated text (or entire tables) together.

3.  **Advanced Retrieve-Rerank Pipeline:** This is a two-stage retrieval process for maximum accuracy.

      * **Retrieve:** A fast vector search (`ChromaDB`) first finds the top 25 *potential* matching documents.
      * **Rerank:** A more powerful `CrossEncoderReranker` (`BAAI/bge-reranker-base`) then re-reads all 25 chunks against the query and picks the *true* top 5, solving the "lost in the middle" problem.

4.  **100% Local & Private:** The entire pipeline runs locally. No API keys, no data leaks.

      * **LLM:** `Ollama` (running `mistral`)
      * **Embeddings:** `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)
      * **Vector Store:** `ChromaDB`

-----

## 💻 Tech Stack

  * **Python 3.11**
  * **Core RAG:** `langchain`, `langchain-classic`, `langchain-core`
  * **Integrations:** `langchain-community`, `langchain-chroma`, `langchain-ollama`, `langchain-huggingface`
  * **Data Ingestion:** `pymupdf4llm`, `langchain-text-splitters`
  * **Embeddings:** `sentence-transformers`
  * **LLM:** `Ollama`
  * **Vector Store:** `ChromaDB`

-----

## 🚀 How to Run This Project

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YourUsername/RAG-Annual-Report.git
    cd RAG-Annual-Report
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    .\venv\Scripts\Activate
    ```

3.  **Install all dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Ollama:**

      * [Install Ollama](https://ollama.com/) on your machine.
      * Pull the Mistral model:
        ```bash
        ollama pull mistral
        ```

5.  **Add Your Data:**

      * Place your PDF file (e.g., `telkom annual reports 2024.pdf`) inside the `/data` folder.
      * *Note: Make sure the filename matches the one in `src/populate_database.py`.*

6.  **Step 1: Populate the Database**

      * Run the ingestion script. This will read the PDF, create embeddings, and build the `chroma` database. This may take a few minutes.
        ```bash
        python -m src.populate_database
        ```
      * *(To clear the database and start fresh, run `python -m src.populate_database --reset`)*

7.  **Step 2: Query Your Data\!**

      * Run the query script with your question in quotes. The first run will download the 1.1GB reranker model.
        ```bash
        python -m src.query_data "What was the total revenue in 2024?"
        ```
        ```bash
        python -m src.query_data "How many subsidiaries does Telkom have?"
        ```

-----

## 📈 Future Improvements

This project is a strong foundation. The next steps to productionalize it would be:

  - [ ] **Build a Web Interface:** Wrap the `query_data.py` logic in a `Streamlit` app to create an interactive "chat-with-your-doc" UI.
  - [ ] **Formal Evaluation:** Create a `test_rag.py` file with a set of questions and expected answers to formally benchmark the accuracy of different retrievers or LLMs.
  - [ ] **Dockerize the Application:** Containerize the entire app for easy and reproducible deployment.
