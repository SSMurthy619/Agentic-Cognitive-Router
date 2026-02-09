# pdf_extractor.py — unified text + OCR extractor for PDFs

import os
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from typing import List
from config import TESSERACT_PATH, POPPLER_PATH, CHUNK_SIZE, CHUNK_OVERLAP

# Configure pytesseract
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

class PDFChunk:
    def __init__(self, text: str, metadata: dict = None):
        self.text = text
        self.metadata = metadata or {}

class PDFExtractor:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def load_all_pdfs_from_folder(self, folder: str) -> List[PDFChunk]:
        chunks = []
        for file in os.listdir(folder):
            if file.lower().endswith(".pdf"):
                path = os.path.join(folder, file)
                chunks.extend(self.extract_chunks(path))
        return chunks

    def extract_chunks(self, pdf_path: str) -> List[PDFChunk]:
        """Try text extraction with PyMuPDF → pdfplumber → OCR fallback."""
        text = self._extract_with_pymupdf(pdf_path)
        if not text.strip():
            text = self._extract_with_pdfplumber(pdf_path)
        if not text.strip():
            text = self._extract_with_ocr(pdf_path)

        return self._chunk_text(text, pdf_path)

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = " ".join([page.get_text("text") for page in doc])
            doc.close()
            return text
        except Exception:
            return ""

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_text = []
                for page in pdf.pages:
                    pages_text.append(page.extract_text() or "")
                return "\n".join(pages_text)
        except Exception:
            return ""

    def _extract_with_ocr(self, pdf_path: str) -> str:
        try:
            images = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
            ocr_text = []
            for img in images:
                text = pytesseract.image_to_string(img)
                ocr_text.append(text)
            return "\n".join(ocr_text)
        except Exception:
            return ""

    def _chunk_text(self, text: str, pdf_path: str) -> List[PDFChunk]:
        """Split long text into overlapping chunks for embeddings."""
        chunks = []
        words = text.split()
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            segment = " ".join(words[i:i + self.chunk_size])
            if segment.strip():
                chunks.append(PDFChunk(segment, {"source": pdf_path}))
        return chunks


# CLI for quick testing
if __name__ == "__main__":
    folder = input("Enter folder path with PDFs: ").strip()
    extractor = PDFExtractor()
    docs = extractor.load_all_pdfs_from_folder(folder)
    print(f"✅ Extracted {len(docs)} chunks")
    for d in docs[:3]:
        print("----")
        print(d.text[:400], "...")
