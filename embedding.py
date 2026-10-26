from langchain_huggingface import HuggingFaceEmbeddings 






def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": 32,
            "normalize_embeddings": True,
        },
        show_progress=True,
    )




def get_qwen_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 16,
        },
        query_encode_kwargs={
            "prompt_name": "query",
            "normalize_embeddings": True,
            "batch_size": 16,
        },
        show_progress=True,
    )


