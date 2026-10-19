from langchain_huggingface import HuggingFaceEmbeddings 



def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
         model_name="Qwen/Qwen3-Embedding-0.6B"
    )
    return embedding_model





