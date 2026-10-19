from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from tokenization import bm25_preprocess

def build_retrievers(  chunked_documents , vector_store ) : 

    chroma_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    bm25_retriever = BM25Retriever.from_documents(chunked_documents, preprocess_func=bm25_preprocess, k=4)
    ensemble = EnsembleRetriever(retrievers=[ bm25_retriever , chroma_retriever])
    return chroma_retriever , bm25_retriever , ensemble 






