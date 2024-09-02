import fitz
import os

def extract_text_from_pdf(pdf_path):
    text_data = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                text_data.append({'page': page_num, 'text': text})
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text_data

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text_data = file.read()
    return text_data

