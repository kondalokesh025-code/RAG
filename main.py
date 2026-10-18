from  giturlparse import parse 
from ingestion import ingest_repository
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
url = "https://github.com/kondalokesh025-code/ANN.git"

p = parse(url)

owner = p.owner 
repo = p.repo 


if   __name__ == "__main__" :
    documents  = ingest_repository(owner , repo )
    chunked_documents = chunk_documents(documents )
    embed_model = get_embedding_model()
    vector_store = create_vector_store(chunked_documents= chunked_documents , embed_model = embed_model , owner= owner , repo = repo  )













