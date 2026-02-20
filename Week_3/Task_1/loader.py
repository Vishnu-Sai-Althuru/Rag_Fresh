from langchain_community.document_loaders import PyPDFLoader

def load_docs():
    loader = PyPDFLoader("data/AI-NOTES-UNIT-1.pdf")
    return loader.load()