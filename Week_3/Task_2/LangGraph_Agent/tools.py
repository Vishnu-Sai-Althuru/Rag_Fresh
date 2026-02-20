from ddgs import DDGS
import sqlite3
import os

# ---------- SEARCH TOOL ----------
def search_tool(query: str):
    results = []
    # with DDGS() as ddgs:
    #     for r in ddgs.text(query, max_results=3):
    #         results.append(r["body"])
    # return "\n".join(results)
    
    with DDGS(verify=False) as ddgs:
        for r in ddgs.text(query, max_results=3):
            results.append(r["body"])
    return "\n".join(results)


# ---------- DATABASE TOOL ----------
def db_query_tool(query: str):

    conn = sqlite3.connect("agent.db")
    cursor = conn.cursor()

    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()
    return str(rows)


# ---------- FILE TOOL ----------
# def file_reader_tool(filename: str):

#     path = os.path.join("files", filename)

#     if not os.path.exists(path):
#         return "File not found"

#     with open(path, "r") as f:
#         return f.read()
import os

def file_reader_tool(filename: str):

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(BASE_DIR, "files", filename)

    if not os.path.exists(path):
        return "File not found"

    with open(path, "r") as f:
        return f.read()