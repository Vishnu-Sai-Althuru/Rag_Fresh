from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class LongTermMemory:
    def __init__(self):
        self.embedding = OllamaEmbeddings(model="nomic-embed-text")
        self.db = Chroma(
            collection_name="memory_store",
            embedding_function=self.embedding,
            persist_directory="./memory_db"
        )

    def add_memory(self, text):
        doc = Document(page_content=text)
        self.db.add_documents([doc])

    def search_memory(self, query):
        results = self.db.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in results])
