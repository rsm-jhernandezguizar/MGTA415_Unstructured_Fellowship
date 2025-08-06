import fitz

file_path = "/home/jovyan/MGTA415_Unstructured_Fellowship/RAG_LLM_DB/Dungeon Master's Guide.pdf"

def extract_text(file_path):
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "file": file_path,
            "page": i + 1,
            "text": text
        })
    return pages