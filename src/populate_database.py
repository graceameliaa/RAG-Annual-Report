import argparse
import os
import shutil
import pymupdf4llm
from langchain_text_splitters import MarkdownTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    # Load the single annual report
    documents = load_report()
    
    # Split the report using Markdown structure
    chunks = split_documents(documents)
    
    # Add to Chroma (this function is now simpler)
    add_to_chroma(chunks)


def load_report():
    """
    Loads the annual report from the data path and converts it to Markdown.
    """
    pdf_path = os.path.join(DATA_PATH, "telkom annual reports 2024.pdf")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return []

    print(f"Loading and converting {pdf_path} to Markdown... (this may take a minute)")
    
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print("Conversion to Markdown complete.")
    
    # Create a single "Document" with the full Markdown content
    # Add the source to know where it came from
    report_doc = Document(
        page_content=md_text,
        metadata={"source": "telkom_annual_report.pdf"}
    )
    return [report_doc] # Return as a list


def split_documents(documents: list[Document]):
    """
    Splits the document based on Markdown headers, tables, and code blocks.
    """
    # Use MarkdownTextSplitter, which is "structure-aware"
    # It will try to keep headers and their paragraphs together.
    # It will also keep tables whole as a single chunk.
    text_splitter = MarkdownTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    
    print("Splitting document based on Markdown structure...")
    chunks = text_splitter.split_documents(documents)
    print(f"Successfully split the report into {len(chunks)} chunks.")
    return chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )

    # Get existing IDs
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks_to_add = []
    new_ids_to_add = []

    # Iterate through all chunks and assign a persistent ID
    for i, chunk in enumerate(chunks):
        # Create a unique, persistent ID, e.g., "data/report.pdf-chunk-0"
        chunk_id = f"{chunk.metadata['source']}-chunk-{i}"
        
        # Add the ID to the metadata *itself*
        # This is the line your query_data.py is looking for!
        chunk.metadata["id"] = chunk_id 
        
        # Now, check if this ID already exists in the DB
        if chunk_id not in existing_ids:
            new_chunks_to_add.append(chunk)
            new_ids_to_add.append(chunk_id)

    if len(new_chunks_to_add):
        print(f"Adding {len(new_chunks_to_add)} new documents...")
        # Add to the database
        db.add_documents(
            documents=new_chunks_to_add,
            ids=new_ids_to_add
        )
    else:
        print("No new documents to add")


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()