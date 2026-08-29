import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our saved ChromaDB
client = chromadb.PersistentClient(
    path="chroma_db"
)


# 3. Open the collection we created in Build 4
collection = client.get_collection(
    name="rag_documents"
)


# 4. Create the retriever function
def retrieve(question, top_k=2):

    question_embedding = model.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context = "\n\n".join(documents)

    return {
        "question": question,
        "context": context,
        "documents": documents,
        "metadatas": metadatas,
        "distances": distances
    }
# 5. TESTING CODE
# This only runs when retriever.py is executed directly
if __name__ == "__main__":

    question = "How many vacation days do employees receive?"

    response = retrieve(question)

    print("\nQuestion:")
    print(response["question"])

    print("\nContext:")
    print(response["context"])

    print("\nSources:")

    for metadata in response["metadatas"]:
        print(metadata)
