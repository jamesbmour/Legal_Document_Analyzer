from langchain_docling.loader import DoclingLoader


loader = DoclingLoader(file_path=FILE_PATH)

docs = loader.load()

print(f"Number of pages: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Page {i + 1} ---\n")
    print(doc.page_content)
    
#%%    
import markitdown
FILE_PATH = "./Beedles 9560B.pdf"

src_file_path: str = FILE_PATH

md = markitdown.MarkItDown()
result = md.convert(src_file_path)
print(result.markdown)
with open("markitdown-poc-output.md", "w", encoding="utf-8") as f:
    f.write(result.markdown)
    
print(result)
# %%
