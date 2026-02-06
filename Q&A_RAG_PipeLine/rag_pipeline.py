from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama


### Load PDF documents using PyPDFLoader
loader = PyPDFLoader("Data/AI-NOTES-UNIT-1.pdf")
documents = loader.load()

### Split documents into smaller chunks using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

### Create embeddings for the chunks using SentenceTransformer
class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

embedding = LocalEmbeddings()

### Store the embeddings in Chroma vector store
vectordb = Chroma.from_documents(chunks, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


### Initialize the Ollama LLM
llm = Ollama(model="llama3")

###  Adding Prompt Directly in the code
question = "What  is AI?"

docs = retriever.get_relevant_documents(question)
context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
You are a factual AI assistant.

Answer ONLY from the context below.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)
print(response)
