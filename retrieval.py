from giturlparse import parse
from ingestion import ingest_repository
from chunking import chunk_documents
from embedding import get_embedding_model
from vector_store import create_vector_store
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever
from tokenization import bm25_preprocess

url = "https://github.com/kondalokesh025-code/ANN.git"
parsed = parse(url)
owner, repo = parsed.owner, parsed.repo

documents = ingest_repository(owner, repo)
chunked_documents = chunk_documents(documents)
embed_model = get_embedding_model()

vector_store = create_vector_store(chunked_documents, embed_model, owner, repo)
chroma_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
bm25_retriever = BM25Retriever.from_documents(chunked_documents, preprocess_func=bm25_preprocess, k=4)


queries = [
    "Where is the preprocessing function implemented?",
    "Where is the neural network model defined, and what layers does it contain?",
    "Where are the optimizer and learning rate defined?",
    "How does the model perform its forward pass?",
    "Where does backpropagation happen in the training loop?",
    "How are features and labels passed to the model?",
    "How is validation accuracy calculated?",
    "How does the training loop save the best model checkpoint?",
    "How is the test set evaluated?",
    "How is the dataset loaded and parsed?",
    "How are the input features preprocessed before training?",
    "How are gas labels mapped to their class names?",
    "How does preprocessing connect to model training?",
    "How does the project evaluate performance across different time batches?",
    "Which files implement the complete pipeline from data loading to evaluation?",
]


ensemble = EnsembleRetriever(retrievers=[ bm25_retriever , chroma_retriever])



if __name__ == "__main__":
    for query in queries:
        results = ensemble([chroma_retriever.invoke(query), bm25_retriever.invoke(query)])

        print(f"\nQuery: {query}")

        for rank, ((source, content), score) in enumerate(results[:4], start=1):
            print(f"\n{rank}. {source} | Score: {score:.4f}")
            print(content[:200])