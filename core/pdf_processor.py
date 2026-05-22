import re
import pdfplumber
from typing import List, Dict, Any, Union, BinaryIO

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text by removing weird characters and extra whitespace.

    Parameters:
        text (str): The raw text extracted from the PDF.

    Returns:
        str: The cleaned and consolidated text.
    """
    if not text:
        return ""
    # Remove non-printable or weird characters but keep punctuation
    cleaned = re.sub(r"[^\x20-\x7E\n\r\t]", "", text)
    # Replace multiple spaces/tabs/newlines with a single space
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def chunk_page_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of a given word size and overlap.

    Parameters:
        text (str): The cleaned text to be chunked.
        chunk_size (int): Max number of words per chunk.
        overlap (int): Number of words overlapping between adjacent chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def extract_pages(pdf_file: Union[str, BinaryIO]) -> List[Dict[str, Any]]:
    """
    Extracts raw text page by page from a PDF file using pdfplumber.

    Parameters:
        pdf_file (Union[str, BinaryIO]): File path or Streamlit file-like object.

    Returns:
        List[Dict[str, Any]]: A list of dicts with 'page_number' and 'text'.
    """
    pages_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_data.append({
                "page_number": idx + 1,
                "text": text
            })
    return pages_data

def process_pdf(pdf_file: Union[str, BinaryIO], filename: str) -> List[Dict[str, Any]]:
    """
    Main function to process a single PDF into list of metadata-enriched chunks.

    Parameters:
        pdf_file (Union[str, BinaryIO]): The PDF file object or path.
        filename (str): The name of the file for metadata storage.

    Returns:
        List[Dict[str, Any]]: List of chunk dicts containing 'text' and 'metadata'.
    """
    raw_pages = extract_pages(pdf_file)
    all_chunks = []
    
    # First pass: Generate all chunks and map metadata
    for page in raw_pages:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue
        page_chunks = chunk_page_text(cleaned)
        for sub_idx, chunk_text_str in enumerate(page_chunks):
            all_chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    "filename": filename,
                    "page_number": page["page_number"],
                    "chunk_index": len(all_chunks)  # Temporary index
                }
            })
            
    # Second pass: Update chunk_index and total_chunks in metadata
    total = len(all_chunks)
    for idx, chunk in enumerate(all_chunks):
        chunk["metadata"]["chunk_index"] = idx + 1
        chunk["metadata"]["total_chunks"] = total
        
    return all_chunks
