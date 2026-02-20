from loader import load_docs
from splitter import custom_split
from vectorstore import create_vectorstore
from rag_chain import build_rag

docs = load_docs()

chunks = custom_split(docs)

vectordb = create_vectorstore(chunks)

retriever, llm = build_rag(vectordb)

question = input("Ask question: ")

retrieved_docs = retriever.invoke(question)

context = "\n".join([d.page_content for d in retrieved_docs])

prompt = f"""
Answer using context only.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n", response)