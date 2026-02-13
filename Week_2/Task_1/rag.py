from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import Document

embedding = OllamaEmbeddings(model="nomic-embed-text")

def create_vector_store(texts):
    docs = [Document(page_content=t) for t in texts]
    vectordb = Chroma.from_documents(docs, embedding)
    return vectordb