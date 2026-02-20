import sqlite3

conn = sqlite3.connect("agent.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS employees(
id INTEGER,
name TEXT,
role TEXT
)
""")

c.execute("INSERT INTO employees VALUES (1,'Vishnu','AI Engineer')")
c.execute("INSERT INTO employees VALUES (2,'Sai','DevOps Engineer')")

conn.commit()
conn.close()