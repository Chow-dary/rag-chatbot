from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("data/sample.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(full_text)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print("\n--- CHUNK", i + 1, "---")
    print(chunk)