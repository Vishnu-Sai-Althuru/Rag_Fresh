from planner import plan_tasks
from executor import execute_task
from retry_handler import execute_with_retry

def orchestrate(user_input):

    print("Planning subtasks...")
    tasks = plan_tasks(user_input)

    results = []

    for task in tasks:
        result = execute_with_retry(execute_task, task)

        if result is None:
            print("Workflow stopped due to failure.")
            return

        results.append(result)

    print("\nAll tasks completed:")
    for r in results:
        print(r)


if __name__ == "__main__":
    user_question = input("Enter complex task: ")
    orchestrate(user_question)