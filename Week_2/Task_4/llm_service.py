import time
from langchain_ollama import OllamaLLM
from logger import log_event
from metrics import record_request, record_error

llm = OllamaLLM(model="llama3")

def ask_llm(prompt):
    start = time.time()

    try:
        response = llm.invoke(prompt)
        latency = time.time() - start

        tokens = len(str(response)) // 4

        record_request(tokens, latency)

        log_event({
            "event": "llm_call",
            "prompt": prompt,
            "tokens": tokens,
            "latency": latency,
            "status": "success"
        })

        return response

    except Exception as e:
        record_error()

        log_event({
            "event": "error",
            "message": str(e)
        })

        return "Error occurred"