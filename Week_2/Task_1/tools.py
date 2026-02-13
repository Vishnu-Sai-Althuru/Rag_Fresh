# from duckduckgo_search import DDGS
from ddgs import DDGS


def web_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            print("Raw result:", r)  # DEBUG
            results.append(r["body"])
    return results
