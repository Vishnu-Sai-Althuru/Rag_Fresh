from llm_service import ask_llm

questions = [
    "What is AI?",
    "Explain ML",
    "What is DevOps?"
]

for q in questions:
    print(ask_llm(q))