from langchain_docling.loader import DoclingLoader

FILE_PATH = "./Beedles 9560B.pdf"

loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()

print(f"Number of pages: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Page {i + 1} ---\n")
    print(doc.page_content)