from langchain_community.llms.ollama import Ollama

class LLMClient:
    def __init__(self):
        self.llm = Ollama(model="llama3")

    def generate(self, prompt):
        return self.llm.invoke(prompt)
