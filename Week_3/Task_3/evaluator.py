# from dataset import DATASET
# from rag_system import rag_answer
# from metrics import semantic_accuracy

from .dataset import DATASET
from .rag_system import rag_answer
from .metrics import semantic_accuracy

def evaluate():

    results = []

    for sample in DATASET:

        question = sample["question"]
        gt = sample["ground_truth"]

        prediction = rag_answer(question)

        metric = semantic_accuracy(gt, prediction)

        results.append({
            "question": question,
            "score": metric["score"],
            "correct": metric["correct"]
        })

    return results