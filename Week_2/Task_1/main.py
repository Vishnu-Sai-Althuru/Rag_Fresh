from tools import web_search
from rag import create_vector_store
from agent import summarize

def run_agent(question):
    print("🔎 Searching web...")
    search_results = web_search(question)

    print("Search Results:", search_results)


    print("📦 Creating vector store...")
    vectordb = create_vector_store(search_results)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)

    context = "\n".join([doc.page_content for doc in docs])

    print("🧠 Generating final answer...")
    answer = summarize(context, question)

    print("\n--- FINAL ANSWER ---\n")
    print(answer)

if __name__ == "__main__":
    user_question = input("Ask a research question: ")
    run_agent(user_question)
