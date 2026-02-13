import time

def execute_with_retry(task_function, task_name, retries=3):
    attempt = 0

    while attempt < retries:
        try:
            result = task_function(task_name)
            return result
        except Exception as e:
            print(f"Error: {e}")
            attempt += 1
            print(f"Retrying {task_name} ({attempt}/{retries})...")
            time.sleep(1)

    print(f"{task_name} permanently failed after retries.")
    return None
