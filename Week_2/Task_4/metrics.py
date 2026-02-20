# metrics = {
#     "requests": 0,
#     "tokens": 0,
#     "errors": 0,
#     "latencies": []
# }

# def record_request(tokens, latency):
#     metrics["requests"] += 1
#     metrics["tokens"] += tokens
#     metrics["latencies"].append(latency)

# def record_error():
#     metrics["errors"] += 1

# def get_metrics():
#     return metrics

import json
import os

# METRICS_FILE = "metrics.json"

# Go to project root (RAG-FRESH)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

METRICS_FILE = os.path.join(BASE_DIR, "metrics.json")

# default metrics
default_metrics = {
    "requests": 0,
    "tokens": 0,
    "errors": 0,
    "latencies": []
}


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    return default_metrics.copy()


def save_metrics(data):
    with open(METRICS_FILE, "w") as f:
        json.dump(data, f, indent=4)


def record_request(tokens, latency):
    data = load_metrics()
    data["requests"] += 1
    data["tokens"] += tokens
    data["latencies"].append(latency)
    save_metrics(data)


def record_error():
    data = load_metrics()
    data["errors"] += 1
    save_metrics(data)


def get_metrics():
    return load_metrics()