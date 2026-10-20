from  giturlparse import parse 
from ingestion import ingest_repository
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
from retrieval import build_retrievers
from eval_retrieval import evaluate_retrieval
import time 
url = "https://github.com/kondalokesh025-code/ANN.git"

p = parse(url)

owner = p.owner 
repo = p.repo 


if   __name__ == "__main__" :
    start_time = time.perf_counter()

    documents = ingest_repository(owner, repo)
    chunked_documents = chunk_documents(documents)
    embed_model = get_embedding_model()

    vector_store = create_vector_store(chunked_documents , embed_model , owner , repo )

    rerank = build_retrievers(chunked_documents, vector_store)

    evaluate_retrieval(chunked_documents , vector_store , rerank )

    end_time = time.perf_counter()
    print(f"\nTotal application runtime: {end_time - start_time:.2f} seconds")















