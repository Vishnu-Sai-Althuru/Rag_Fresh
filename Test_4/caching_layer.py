from functools import lru_cache

@lru_cache(maxsize=100)
def cached_llm_call(prompt):
    from llm_client import LLMClient
    client = LLMClient()
    print("LLM Executed")
    return client.generate(prompt)
