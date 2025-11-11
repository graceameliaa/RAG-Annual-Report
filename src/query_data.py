import argparse
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # 1. Prepare the DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # 2. Create the new "Retrieve-Rerank" pipeline
    
    # Create the base retriever, and ask it for a large number of chunks
    base_retriever = db.as_retriever(search_kwargs={"k": 25})

    # Initialize the reranker model
    model = HuggingFaceCrossEncoder(
        model_name="BAAI/bge-reranker-base",
        model_kwargs={'device': 'cpu'} 
    )

    # Initialize the reranker with the model,
    # and tell it to return the new top 5 documents.
    compressor = CrossEncoderReranker(model=model, top_n=5)

    # Create the final compression retriever
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
    )

    print("Retrieving and reranking documents...")
    
    # 3. Use the new retriever to get the most relevant documents
    results = compression_retriever.invoke(query_text)

    # 4. Format the context and generate the answer
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text


if __name__ == "__main__":
    main()