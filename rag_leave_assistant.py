import requests
import chromadb


# -----------------------------------------
# 1. Connect to Vector DB
# -----------------------------------------

client = chromadb.PersistentClient(
    path="./leave_policy_db"
)

collection = client.get_collection(
    name="leave_policy"
)


# -----------------------------------------
# 2. Create embedding
# -----------------------------------------

def create_embedding(text):

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    return response.json()["embedding"]


# -----------------------------------------
# 3. Ask user
# -----------------------------------------

question = input("Ask HR Leave Policy question: ")


# -----------------------------------------
# 4. Convert question → vector
# -----------------------------------------

query_embedding = create_embedding(question)


# -----------------------------------------
# 5. Retrieve relevant chunks
# -----------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


relevant_chunks = results["documents"][0]


# -----------------------------------------
# 6. Build Context
# -----------------------------------------

context = "\n\n".join(relevant_chunks)


# -----------------------------------------
# 7. Create Prompt
# -----------------------------------------

prompt = f"""
You are an HR policy assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer is not present in the context,
say:

"I could not find this information in the
company leave policy."

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


# -----------------------------------------
# 8. Call Ollama
# -----------------------------------------

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
)


# -----------------------------------------
# 9. Get answer
# -----------------------------------------

answer = response.json()["message"]["content"]


# -----------------------------------------
# 10. Display
# -----------------------------------------

print("\n===== AI ANSWER =====\n")

print(answer)