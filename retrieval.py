from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from tokenization import bm25_preprocess
from sentence_transformers import CrossEncoder


def build_retrievers(  chunked_documents , vector_store ) : 


    chroma_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(chunked_documents, preprocess_func=bm25_preprocess, k = 20 )
    ensemble = EnsembleRetriever(retrievers=[ bm25_retriever , chroma_retriever])
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(query):
        documents = ensemble.invoke(query)[:20]
        scores = reranker.predict([(query, doc.page_content) for doc in documents])
        ranked = sorted(zip(documents, scores),key=lambda item: item[1],reverse=True)
        return [doc for doc, score in ranked[:5]]

    return rerank 






