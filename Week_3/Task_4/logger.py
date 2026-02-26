import logging
import json
from datetime import datetime

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(message)s"
)

def log_event(event_type, data):
    log_entry = {
        "timestamp": str(datetime.now()),
        "event": event_type,
        "data": data
    }

    logging.info(json.dumps(log_entry))