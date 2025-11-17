from sentence_transformers import SentenceTransformer, util

# Load Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Static Q&A Data
qa_data = {
    "who created python": "Python was created by Guido van Rossum.",
    "who is guido van rossum": "Guido van Rossum is the creator of Python.",
    "what is python": "Python is a programming language.",
}

questions = list(qa_data.keys())
question_embeddings = model.encode(questions)

def get_best_answer(query):
    query_embedding = model.encode(query)
    scores = util.cos_sim(query_embedding, question_embeddings)[0]

    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    return questions[best_idx] if best_score > 0.50 else None


# --- Chatbot Loop ---
print("Chatbot is ready! Type 'exit' to stop.")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    best_question = get_best_answer(query)

    if best_question:
        print("Bot:", qa_data[best_question])
    else:
        print("Bot: Sorry, I don't know the answer.")
