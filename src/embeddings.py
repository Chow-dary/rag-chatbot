from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. Load PDF
reader = PdfReader("data/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(full_text)

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Create embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# 5. Ask a question
question = "How many vacation days do employees receive?"

# 6. Create embedding for the question
question_embedding = model.encode(question)

# 7. Compare question with every chunk
scores = cos_sim(question_embedding, chunk_embeddings)[0]

# 8. Find the highest score
best_index = scores.argmax().item()

# 9. Print result
print("Question:")
print(question)

print("\nBest matching chunk:")
print(chunks[best_index])

print("\nSimilarity score:")
print(scores[best_index].item())