from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

def rag_answer(question):
    prompt = f"""
Answer briefly:

Question: {question}
"""
    return llm.invoke(prompt)