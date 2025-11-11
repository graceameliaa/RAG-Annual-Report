from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Gets a free, local embedding function from Hugging Face.
    """
    # This will download the model "all-MiniLM-L6-v2" (about 80MB) 
    # the first time you run it and store it in a cache.
    model_name = "all-MiniLM-L6-v2" 
    
    # We set the model to run on your CPU. 
    # If you have a powerful GPU, you could change 'cpu' to 'cuda'.
    model_kwargs = {'device': 'cpu'}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    
    print("Embedding function loaded from Hugging Face (all-MiniLM-L6-v2).")
    return embeddings

# You can run this file directly to test it: python get_embedding_function.py
if __name__ == "__main__":
    try:
        embeddings = get_embedding_function()
        test_vector = embeddings.embed_query("This is a test sentence.")
        print(f"Test vector length: {len(test_vector)}")
        print("Success! Hugging Face embeddings are working.")
    except Exception as e:
        print(f"An error occurred: {e}")