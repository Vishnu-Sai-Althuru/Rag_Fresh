import time
from langchain_ollama import OllamaLLM
from logger import log_event
from metrics import update_metrics

llm = OllamaLLM(model="llama3")

def ask_llm(prompt):
    start = time.time()

    try:
        log_event("request", {"prompt": prompt})

        response = llm.invoke(prompt)

        latency = time.time() - start
        tokens = len(prompt + response) // 4

        update_metrics(tokens, latency)

        log_event("response", {"response": response})

        return response

    except Exception as e:
        update_metrics(0, 0, error=True)
        log_event("error", {"message": str(e)})
        return "Error occurred"