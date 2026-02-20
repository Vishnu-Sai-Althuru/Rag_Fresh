from agent import app

while True:
    question = input("Ask: ")

    result = app.invoke({"question": question})

    print("\nAnswer:\n", result["answer"])