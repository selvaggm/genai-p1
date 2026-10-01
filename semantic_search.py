import requests
import chromadb


# -----------------------------------------
# 1. Connect to ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(
    path="./leave_policy_db"
)

collection = client.get_collection(
    name="leave_policy"
)


# -----------------------------------------
# 2. Embedding function
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
# 3. User question
# -----------------------------------------

question = input("Ask your question: ")


# -----------------------------------------
# 4. Convert question into vector
# -----------------------------------------

query_embedding = create_embedding(question)


# -----------------------------------------
# 5. Semantic search
# -----------------------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)


# -----------------------------------------
# 6. Display results
# -----------------------------------------

print("\n===== RELEVANT INFORMATION =====\n")

for document in results["documents"][0]:

    print(document)
    print("-" * 60)