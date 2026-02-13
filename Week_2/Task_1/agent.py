from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

# def summarize(context, question):
#     prompt = f"""
# You are a research assistant 

# Answer the question using only the context below.

# Context: 
# {context}

# Question:
# {question}

# Provide a clear summarized answer.
# """
#     return llm.invoke(prompt)

def summarize(context, question):
    prompt = f"""
You are a research assistant.

You may use:
1. The provided context.

If the context is empty, answer using your knowledge.

Context:
{context}

Question:
{question}

Provide a clear answer.
"""
    return llm.invoke(prompt)
