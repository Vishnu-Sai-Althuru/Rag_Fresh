import time
import json
from cost_tracker import CostTracker
from caching_layer import cached_llm_call
from batching_layer import batch_prompts

tracker = CostTracker()

questions = [
    "What is AI?",
    "Explain Machine Learning.",
    "What is AI?"  # duplicate for caching test
]

print("---- WITHOUT BATCHING ----")

for q in questions:
    prompt = f"Answer briefly: {q}"

    start = time.time()
    response = cached_llm_call(prompt)
    end = time.time()

    usage = tracker.track(prompt, response)

    print("Usage:", usage)
    print("Latency:", round(end - start, 2), "seconds\n")

print("---- SUMMARY ----")
summary = tracker.summary()
print(summary)

with open("cost_report.json", "w") as f:
    json.dump(summary, f, indent=4)
