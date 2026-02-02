import uuid
from io import BytesIO
from typing import List, Optional
import re
import unicodedata

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# OCR dependencies (will be imported conditionally)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  OCR libraries not installed. Install with: pip install pdf2image pytesseract")


# =========================
# ChromaDB Configuration
# =========================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR
    )
)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================
# Collection Helper
# =========================

def get_or_create_collection():
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return collection


# =========================
# Text Normalization (HIGH ROI)
# =========================

def normalize_text(text: str) -> str:
    """
    Normalize text for better processing:
    - Unicode normalization
    - Fix broken hyphenation
    - Collapse excessive whitespace
    """
    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    
    # Fix broken hyphenation (word- \n word → word-word)
    text = text.replace("-\n", "")
    
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


# =========================
# OCR Fallback (BIGGEST WIN)
# =========================

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """
    Extract text using OCR (Tesseract).
    Used for scanned PDFs where pypdf fails.
    """
    if not OCR_AVAILABLE:
        raise ImportError("OCR libraries not installed. Install: pip install pdf2image pytesseract poppler-utils")
    
    try:
        images = convert_from_bytes(pdf_bytes)
        text = ""
        
        for i, img in enumerate(images):
            print(f"  OCR processing page {i+1}/{len(images)}...")
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
        
        return text
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return ""


# =========================
# PDF Text Extraction (WITH OCR FALLBACK)
# =========================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF. Falls back to OCR if needed.
    
    Strategy:
    1. Try pypdf extraction (fast, works for text PDFs)
    2. If text is too short (<300 chars), likely scanned → use OCR
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    text = ""

    # Try standard extraction first
    for i, page in enumerate(reader.pages):
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    # OCR fallback if text is too small (likely scanned PDF)
    if len(text.strip()) < 300:
        print("⚠️  Low text detected. Falling back to OCR...")
        
        if OCR_AVAILABLE:
            try:
                ocr_text = extract_text_with_ocr(pdf_bytes)
                if ocr_text.strip():
                    print(f"✅ OCR successful: {len(ocr_text)} characters extracted")
                    text += "\n" + ocr_text
                else:
                    print("⚠️  OCR returned no text")
            except Exception as e:
                print(f"❌ OCR failed: {e}")
        else:
            print("⚠️  OCR not available. Install: pip install pdf2image pytesseract")
    
    # Normalize the extracted text
    normalized_text = normalize_text(text)
    
    print(f"📄 PDF extraction complete: {len(normalized_text)} characters")
    
    return normalized_text


# =========================
# Chunking (Improved)
# =========================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into word-based chunks with overlap.
    Improved to handle edge cases better.
    """
    words = text.split()
    
    if not words:
        return []
    
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk and len(chunk.strip()) > 0:
            chunks.append(chunk)
    
    print(f"📝 Created {len(chunks)} chunks from text")
    
    return chunks


# =========================
# Embeddings
# =========================

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for text chunks using sentence-transformers.
    """
    if not texts:
        return []
    
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# =========================
# Store PDF in ChromaDB
# =========================

def embed_and_store_pdf(pdf_text: str, report_id: str) -> int:
    """
    Chunk PDF text, generate embeddings, and store in ChromaDB.

    Returns number of chunks indexed.
    """
    # Normalize first
    normalized_text = normalize_text(pdf_text)
    
    # Chunk
    chunks = chunk_text(normalized_text)

    # Safety check
    if not chunks:
        print("⚠️  No chunks created from PDF text")
        return 0

    # Embed
    embeddings = generate_embeddings(chunks)

    # Collection
    collection = get_or_create_collection()

    # IDs
    chunk_ids = [f"{report_id}_chunk_{i}" for i in range(len(chunks))]

    # Metadata
    metadatas = [
        {
            "report_id": report_id,
            "chunk_index": i,
            "total_chunks": len(chunks)
        }
        for i in range(len(chunks))
    ]

    # Store (Chroma auto-persists in modern versions)
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )

    print(f"✅ Stored {len(chunks)} chunks for report {report_id}")

    return len(chunks)


# =========================
# RAG Retrieval (Case-Isolated)
# =========================

def retrieve_context(query: str, report_id: str, top_k: int = 5):
    """
    Retrieve most relevant PDF chunks for a specific report_id.
    Prevents cross-case contamination (CRITICAL for PV systems).
    """
    collection = get_or_create_collection()

    # Normalize query
    normalized_query = normalize_text(query)
    
    # Embed query
    query_embedding = generate_embeddings([normalized_query])[0]

    # Query Chroma WITH report_id filter
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"report_id": report_id}
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    print(f"🔍 Retrieved {len(documents)} chunks for report {report_id}")

    return documents, metadatas


# =========================
# Optional: Clear Vector DB
# =========================

def clear_pdf_collection():
    """
    Clear all stored PDF embeddings.
    """
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("🗑️  PDF collection cleared")
    except Exception as e:
        print(f"⚠️  Failed to clear collection: {e}")
