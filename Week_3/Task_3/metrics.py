from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def semantic_accuracy(expected, predicted, threshold=0.80):

    emb1 = model.encode([expected])
    emb2 = model.encode([predicted])

    score = cosine_similarity(emb1, emb2)[0][0]

    return {
        "score": float(score),
        "correct": score >= threshold
    }