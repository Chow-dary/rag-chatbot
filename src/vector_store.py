import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# 1. Load the PDF
reader = PdfReader("data/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text

# 2. Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(full_text)

print("Number of chunks:", len(chunks))

# 3. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# 5. Create ChromaDB client
client = chromadb.PersistentClient(
    path="chroma_db"
)

# 6. Create a collection
collection = client.get_or_create_collection(
    name="rag_documents"
)

# 7. Create unique IDs for each chunk
ids = []

for i in range(len(chunks)):
    ids.append(f"chunk_{i + 1}")

# 8. Create metadata for each chunk
metadatas = []

for i in range(len(chunks)):
    metadatas.append({
        "source": "sample.pdf",
        "chunk_number": i + 1
    })

# 9. Store chunks + embeddings + metadata
collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=chunk_embeddings.tolist(),
    metadatas=metadatas
)

# 10. Ask a question
question = "How many vacation days do employees receive?"

# 11. Create embedding for the question
question_embedding = model.encode(question)

# 12. Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)

print("\nQuestion:")
print(question)

print("\nTop matching chunks:")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(document)

    print("Metadata:")
    print(results["metadatas"][0][i])

    print("Distance:")
    print(results["distances"][0][i])