from langchain_ollama import OllamaLLM

def build_rag(vectordb):

    retriever = vectordb.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {"source": "AI-NOTES-UNIT-1.pdf"}
        }
    )

    llm = OllamaLLM(model="llama3")

    return retriever, llm