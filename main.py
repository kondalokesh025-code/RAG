from  giturlparse import parse 
from ingestion import ingest_repository
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
from retrieval import build_retrievers
from eval_retrieval import evaluate_retrieval
from generation import generate
import time 

url = "https://github.com/pallets/click.git"


p = parse(url)

owner = p.owner 
repo = p.repo 

if   __name__ == "__main__" :
    start_time = time.perf_counter()

    documents = ingest_repository(owner, repo)
    chunked_documents = chunk_documents(documents)
    embed_model = get_embedding_model()

    vector_store = create_vector_store(chunked_documents , embed_model , owner , repo )

    rerank  = build_retrievers(chunked_documents, vector_store)

    query = "what database does this repository use ? "

    # Retrieve relevant chunks
    retrieval_start = time.perf_counter()
    chunks = rerank(query)
    retrieval_time = time.perf_counter() - retrieval_start

    # Generate an answer
    generation_start = time.perf_counter()
    answer = generate(query, chunks)
    generation_time = time.perf_counter() - generation_start

    print("\n========== ANSWER ==========")
    print(answer)

    print(f"\nRetrieval time: {retrieval_time:.2f} seconds")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Total application runtime: {time.perf_counter() - start_time:.2f} seconds")
    print( evaluate_retrieval(chunked_documents , vector_store , rerank ))


















