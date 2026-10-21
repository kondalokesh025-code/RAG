from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def generate(query , chunks  ) : 
    model = ChatOllama(model =  "qwen3:4b-instruct-2507-q4_K_M" , temperature = 0  )
    parser = StrOutputParser()
    prompt = PromptTemplate.from_template(
    """Answer the question using the repository context below.
    Question: {query}
    Repository context:
    {chunks}
    If the context does not contain the answer, say:
    "The provided repository context does not contain enough information to answer."
    """
    )
    context = "\n\n".join(
    f"Source: {doc.metadata.get('source', 'Unknown')}\n"
    f"Content:\n{doc.page_content}"
    for doc in chunks
    )
    chain  = prompt|model|parser 
    return chain.invoke({"query" : query , "chunks" : context  })