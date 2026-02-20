import logging
import json

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(message)s"
)

def log_event(data: dict):
    logging.info(json.dumps(data))