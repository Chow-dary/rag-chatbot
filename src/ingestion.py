from pypdf import PdfReader
reader = PdfReader("data/sample.pdf")
full_text = ""
for page in reader.pages:
    text = page.extract_text()
    full_text = full_text + text
print(full_text)
