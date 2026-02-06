from metrics import semantic_correctness, citation_grounded, toxicity_score

def evaluate(entry):
    expected = entry["expected_answer"]
    answer = entry["model_answer"]
    context = entry["context"]

    # Correctness
    is_correct, similarity = semantic_correctness(expected, answer)

    # Citation
    citation = citation_grounded(answer, context)

    # Toxicity
    tox_score = toxicity_score(answer)

    return {
        "correctness": is_correct,
        "similarity_score": round(similarity, 3),
        "citation_grounded": citation,
        "toxicity_score": round(tox_score, 3)
    }
