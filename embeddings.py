import requests
import chromadb
from pypdf import PdfReader


# -----------------------------------------
# 1. Read PDF
# -----------------------------------------

pdf_file = "sample_company_leave_policy.pdf"

reader = PdfReader(pdf_file)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"


# -----------------------------------------
# 2. Create simple chunks
# -----------------------------------------

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

print("Number of chunks:", len(chunks))


# -----------------------------------------
# 3. Create embedding function
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
# 4. Create ChromaDB
# -----------------------------------------

client = chromadb.PersistentClient(
    path="./leave_policy_db"
)

collection = client.get_or_create_collection(
    name="leave_policy"
)


# -----------------------------------------
# 5. Generate embeddings
# -----------------------------------------

for index, chunk in enumerate(chunks):

    embedding = create_embedding(chunk)

    collection.add(
        ids=[str(index)],
        documents=[chunk],
        embeddings=[embedding]
    )

    print(f"Stored chunk {index}")


print("\nEmbedding + Vector DB creation completed.")