from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def create_vectorstore(chunks):

    embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="db"
    )

    return vectordb