from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3")

def generate_response(question, short_memory, long_memory):

    # Get conversation history
    chat_context = short_memory.get_context()

    # Get relevant long-term memory
    memory_context = long_memory.search_memory(question)

    prompt = f"""
You are an intelligent assistant.

Conversation History:
{chat_context}

Relevant Long-Term Memory:
{memory_context}

User Question:
{question}

Answer clearly.
"""

    return llm.invoke(prompt)
