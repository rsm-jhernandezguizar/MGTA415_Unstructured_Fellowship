pdf_path = "/home/jovyan/MGTA415_Unstructured_Fellowship/RAG_LLM_DB/Dungeon Master's Guide.pdf"
output_dir = "/home/jovyan/MGTA415_Unstructured_Fellowship/RAG_LLM_DB/images"

def extract_images(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        images = doc[page_index].get_images(full=True)
        for img_index, img in enumerate(images):
            xref = images[img_index][0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = f"{output_dir}/{page_index}_{img_index}.png"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)