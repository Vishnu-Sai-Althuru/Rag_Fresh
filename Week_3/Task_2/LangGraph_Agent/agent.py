from langgraph.graph import StateGraph, END
from typing import TypedDict
from langchain_ollama import OllamaLLM

from tools import search_tool, db_query_tool, file_reader_tool

llm = OllamaLLM(model="llama3")


# -------- Agent State --------
# class AgentState(TypedDict):
#     question: str
#     answer: str
class AgentState(TypedDict):
    question: str
    tool: str
    answer: str


# -------- Decision Node --------
# def decide_tool(state):

#     question = state["question"]

#     prompt = f"""
# You are an agent.

# Choose tool:
# - search
# - db
# - file

# Question: {question}

# Return ONLY tool name.
# """

#     tool = llm.invoke(prompt).lower()

#     return {"tool": tool}

def decide_tool(state):

    question = state["question"]


    prompt = f"""
You are an AI agent.

Choose ONE tool:

search → for general knowledge questions
db → for employee or database questions
file → when user asks to read a file or notes

Question: {question}

Return ONLY one word:
search
db
file
"""

#     prompt = f"""
# Choose tool:
# search, db, file

# Question: {question}
# Return only tool name.
# """


    tool = llm.invoke(prompt).strip().lower()

    return {"tool": tool}

# -------- Tool Executor --------
# def run_tool(state):

#     q = state["question"]

#     if "search" in state["tool"]:
#         result = search_tool(q)

#     elif "db" in state["tool"]:
#         result = db_query_tool("SELECT * FROM employees")

#     elif "file" in state["tool"]:
#         result = file_reader_tool("notes.txt")

#     else:
#         result = "No tool matched"

#     return {"answer": result}


# def run_tool(state):

#     tool = state.get("tool", "")

#     if "search" in tool:
#         result = search_tool(state["question"])

#     elif "db" in tool:
#         result = db_query_tool("SELECT * FROM employees")

#     elif "file" in tool:
#         result = file_reader_tool("notes.txt")

#     else:
#         result = "No tool selected"

#     return {"answer": result}

def run_tool(state):

    tool = state.get("tool", "")
    question = state.get("question", "")

    if "search" in tool:
        result = search_tool(question)

    elif "db" in tool:
        result = db_query_tool("SELECT * FROM employees")

    elif "file" in tool:
        result = file_reader_tool("/Users/vishnusaialthuru/Desktop/BT/rag-fresh/Week_3/Task_2/files/notes.txt")

    else:
        result = "Unknown tool"

    return {"answer": result}


# -------- Build Graph --------
workflow = StateGraph(AgentState)

workflow.add_node("decide", decide_tool)
workflow.add_node("tool_runner", run_tool)

workflow.set_entry_point("decide")
workflow.add_edge("decide", "tool_runner")
workflow.add_edge("tool_runner", END)

app = workflow.compile()