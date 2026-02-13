import random
import time

def execute_task(task_name):
    print(f"Executing: {task_name}")

    # Simulate random failure
    if random.random() < 0.3:
        raise Exception(f"{task_name} failed!")

    time.sleep(1)
    return f"{task_name} completed successfully"