from llm_service import ask_llm

questions = [
    "What is AI?",
    "Explain Machine Learning",
    "What is Deep Learning?"
]

for q in questions:
    print("\nQ:", q)
    print("A:", ask_llm(q))