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
        selected = []
        source_counts = {}
        for doc, score in ranked:
            source = doc.metadata.get("source", "Unknown")
            if source_counts.get(source, 0) >= 2:
                continue
            selected.append(doc)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) == 5:
                break
        return selected 
    return rerank 






