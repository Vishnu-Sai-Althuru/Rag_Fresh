import json
import os

METRIC_FILE = "metrics.json"

def load_metrics():
    if not os.path.exists(METRIC_FILE):
        return {"requests": 0, "tokens": 0, "errors": 0, "latencies": []}

    with open(METRIC_FILE) as f:
        return json.load(f)

def save_metrics(metrics):
    with open(METRIC_FILE, "w") as f:
        json.dump(metrics, f, indent=4)

def update_metrics(tokens, latency, error=False):
    metrics = load_metrics()

    metrics["requests"] += 1
    metrics["tokens"] += tokens
    metrics["latencies"].append(latency)

    if error:
        metrics["errors"] += 1

    save_metrics(metrics)