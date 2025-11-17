import mysql.connector
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
import re

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "timetable_system"
}

GROQ_API_KEY = ""


# -------------------------------------------------------
# LLM (Groq with LLaMA-3)
# -------------------------------------------------------
client = Groq(api_key=GROQ_API_KEY)

def groq_chat(prompt):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Correct access → use dot notation
        return response.choices[0].message.content

    except Exception as e:
        return f"Groq API Error: {str(e)}"


# -------------------------------------------------------
# READ FULL DATABASE
# -------------------------------------------------------
def read_database():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    cursor.execute("SHOW TABLES")
    tables = [t[0] for t in cursor.fetchall()]

    data = []
    for table in tables:
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

        for row in rows:
            text = f"Table {table} → " + "; ".join(f"{cols[i]}: {row[i]}" for i in range(len(cols)))
            data.append(text)

    cursor.close()
    conn.close()
    return data


# -------------------------------------------------------
# BUILD EMBEDDING INDEX
# -------------------------------------------------------
def build_index(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, embeddings


def semantic_search(query, model, index, embeddings, texts):
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    D, I = index.search(q_emb, 5)

    results = [(D[0][i], texts[I[0][i]]) for i in range(len(I[0]))]
    return results


# -------------------------------------------------------
# INTENT DETECTION FOR SQL
# -------------------------------------------------------
def detect_sql_intent(q):
    q = q.lower()

    # COUNT — supports: how many teachers? how many subjects are there?
    m = re.search(r"how many (.+?)(\?|$)", q)
    if m:
        table = m.group(1).strip().replace(" ", "_")
        return f"SELECT COUNT(*) FROM {table};"

    # LIST ALL
    m = re.search(r"list all (.+)", q)
    if m:
        table = m.group(1).strip().replace(" ", "_")
        return f"SELECT * FROM {table};"

    return None


def run_sql_query(query):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]

        cursor.close()
        conn.close()

        return rows, cols

    except Exception as e:
        return None, None


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
print("Reading database...")
texts = read_database()
print(f"Loaded {len(texts)} rows.")

print("Building embeddings...")
model, index, embeddings = build_index(texts)

print("\nChatbot ready! Type your questions (exit to quit)\n")

while True:
    q = input("You: ").strip()
    if q == "exit":
        break

    # 1️⃣ SQL intent detection
    sql = detect_sql_intent(q)
    if sql:
        rows, cols = run_sql_query(sql)
        if rows is not None:
            print("\nSQL Result:")
            for r in rows:
                print(dict(zip(cols, r)))
            print()
            continue

    # 2️⃣ Semantic Search
    results = semantic_search(q, model, index, embeddings, texts)

    context = "\n".join([r[1] for r in results])

    # 3️⃣ Groq reasoning using DB context
    prompt = f"""
Use only the following database records to answer the user question:

{context}

User question: {q}
"""

    answer = groq_chat(prompt)
    print("\nBot:", answer, "\n")
