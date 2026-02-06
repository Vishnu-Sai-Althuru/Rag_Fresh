from eval_dataset import eval_data
from evaluator import evaluate

for i, item in enumerate(eval_data):
    print(f"\n--- Sample {i+1} ---")
    result = evaluate(item)
    print(result)