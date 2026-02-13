#----------------------------------------------------------------------------------------------------
# DESCRIPTION: Enhanced RAG workflow with semantic chunking and deduplication for pharmacovigilance
#              reports. Implements sentence-aware semantic chunking, embedding-based deduplication, and 
#              report-level isolation for retrieval.
#----------------------------------------------------------------------------------------------------
import uuid
from io import BytesIO
from typing import List, Optional, Tuple, Dict
import re
import unicodedata
import numpy as np

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import nltk
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data for sentence tokenization (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# OCR dependencies (conditional import)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print(" OCR libraries not installed. Install with: pip install pdf2image pytesseract")


# =========================
# Configuration
# =========================

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_documents"

# Semantic Chunking Parameters
MAX_CHUNK_SIZE = 500  # Maximum words per chunk
MIN_CHUNK_SIZE = 100  # Minimum words per chunk
OVERLAP_SENTENCES = 2  # Number of sentences to overlap

# Deduplication Parameters
SIMILARITY_THRESHOLD = 0.88  # Cosine similarity threshold (0-1)
                             # Higher = stricter deduplication


# =========================
# ChromaDB Client
# =========================

chroma_client = chromadb.Client(
    Settings(
        persist_directory=CHROMA_DIR
    )
)

# =========================
# Embedding Model
# =========================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ================================================================
# fetches an existing ChromaDB collection or creates it if missing, 
# using cosine similarity for vector search
# ================================================================

def get_or_create_collection():
    
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f" Using existing collection: {COLLECTION_NAME}")
    except Exception:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f" Created new collection: {COLLECTION_NAME}")
    return collection


# ==============================================================
# It cleans and standardizes raw text by normalizing Unicode, 
# fixing line-break hyphenation, removing control characters
# ==============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for better processing:
    - Unicode normalization (NFKD)
    - Fix broken hyphenation across lines
    - Collapse excessive whitespace
    - Remove control characters
    """
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    
    # Fix broken hyphenation (word- \n word → wordword or word-word)
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
    
    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)  # Multiple newlines → single
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces → single
    text = re.sub(r'\n ', '\n', text)  # Remove space after newline
    
    return text.strip()


# ======================================================================
# converts a scanned PDF into images and runs Tesseract OCR on each page
# to extract text when normal PDF parsing fails
# ======================================================================

def extract_text_with_ocr(pdf_bytes: bytes) -> str:
    """
    Extract text using OCR (Tesseract).
    Used for scanned PDFs where pypdf fails.
    """
    if not OCR_AVAILABLE:
        raise ImportError(
            "OCR libraries not installed. Install with: "
            "pip install pdf2image pytesseract pillow"
        )
    
    try:
        print("  Converting PDF pages to images...")
        images = convert_from_bytes(pdf_bytes, dpi=300)
        text = ""
        
        for i, img in enumerate(images, 1):
            print(f"   OCR processing page {i}/{len(images)}...")
            page_text = pytesseract.image_to_string(img, lang='eng')
            text += page_text + "\n\n"
        
        print(f" OCR completed: {len(text)} characters extracted")
        return text
    except Exception as e:
        print(f" OCR extraction failed: {e}")
        return ""


# ================================================================================
# extracts text from a PDF using standard parsing first, falls back to OCR
# if the extracted text is too short, then cleans and normalizes the final output
# ================================================================================

def extract_text_from_pdf(pdf_bytes: bytes, ocr_threshold: int = 300) -> str:
    """
    Extract text from PDF with intelligent OCR fallback.
    
    Strategy:
    1. Try pypdf extraction (fast, works for text PDFs)
    2. If text length < ocr_threshold, assume scanned → use OCR
    
    Args:
        pdf_bytes: PDF file as bytes
        ocr_threshold: Character count below which OCR is triggered
    
    Returns:
        Extracted and normalized text
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        text = ""
        
        print(f" Extracting text from {len(reader.pages)} pages...")

        # Try standard extraction
        for i, page in enumerate(reader.pages, 1):
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n\n"
        
        print(f" Standard extraction: {len(text)} characters")

        # OCR fallback if text is too short (likely scanned)
        if len(text.strip()) < ocr_threshold:
            print(f"  Text length ({len(text)} chars) below threshold ({ocr_threshold}). Initiating OCR...")
            
            if OCR_AVAILABLE:
                try:
                    ocr_text = extract_text_with_ocr(pdf_bytes)
                    if len(ocr_text.strip()) > len(text.strip()):
                        print(f" OCR yielded better results. Using OCR text.")
                        text = ocr_text
                    else:
                        print("  OCR did not improve extraction. Using standard text.")
                except Exception as e:
                    print(f" OCR failed: {e}. Using standard extraction.")
            else:
                print(" OCR not available. Install: pip install pdf2image pytesseract")

#================================================
        # Normalize (Clean up messy text)
#================================================

        normalized_text = normalize_text(text)
        
        print(f"PDF extraction complete: {len(normalized_text)} characters")
        return normalized_text
        
    except Exception as e:
        print(f" PDF extraction error: {e}")
        return ""


# =========================================================================
# Semantic Chunking (: Break long reports into smaller, meaningful pieces)
# =========================================================================

def semantic_chunk_text(
    text: str, 
    max_chunk_size: int = MAX_CHUNK_SIZE, 
    min_chunk_size: int = MIN_CHUNK_SIZE,
    overlap_sentences: int = OVERLAP_SENTENCES
) -> List[Dict[str, any]]:
    """
    Semantic chunking: splits text by sentences and groups intelligently.
    
    Unlike fixed-size chunking, this method:
    - Respects sentence boundaries (no mid-sentence cuts)
    - Groups sentences until max_chunk_size is reached
    - Maintains overlap for context continuity
    - Merges tiny chunks to avoid fragments
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum words per chunk
        min_chunk_size: Minimum words per chunk (merge if smaller)
        overlap_sentences: Number of sentences to carry forward
    
    Returns:
        List of dicts with 'text', 'start_sentence', 'end_sentence'
    """
    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        print(" No sentences found in text")
        return []
    
    print(f" Detected {len(sentences)} sentences")
    
    chunks = []
    current_chunk_sentences = []
    current_word_count = 0
    sentence_idx = 0
    
    while sentence_idx < len(sentences):
        sentence = sentences[sentence_idx]
        sentence_words = len(sentence.split())
        
        # Check if adding this sentence would exceed max_chunk_size
        if current_word_count + sentence_words > max_chunk_size and current_chunk_sentences:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append({
                'text': chunk_text,
                'start_sentence': sentence_idx - len(current_chunk_sentences),
                'end_sentence': sentence_idx - 1,
                'word_count': current_word_count
            })
            
            # Prepare next chunk with overlap
            if overlap_sentences > 0 and len(current_chunk_sentences) > overlap_sentences:
                # Keep last N sentences for context
                current_chunk_sentences = current_chunk_sentences[-overlap_sentences:]
                current_word_count = sum(len(s.split()) for s in current_chunk_sentences)
            else:
                current_chunk_sentences = []
                current_word_count = 0
            
            # Don't increment sentence_idx - we'll process it in next iteration
            continue
        
        # Add sentence to current chunk
        current_chunk_sentences.append(sentence)
        current_word_count += sentence_words
        sentence_idx += 1
    
    # Handle final chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        
        # Merge small final chunks
        if chunks and current_word_count < min_chunk_size:
            print(f" Merging small final chunk ({current_word_count} words) with previous")
            chunks[-1]['text'] += " " + chunk_text
            chunks[-1]['end_sentence'] = len(sentences) - 1
            chunks[-1]['word_count'] += current_word_count
        else:
            chunks.append({
                'text': chunk_text,
                'start_sentence': sentence_idx - len(current_chunk_sentences),
                'end_sentence': len(sentences) - 1,
                'word_count': current_word_count
            })
    
    print(f" Created {len(chunks)} semantic chunks from {len(sentences)} sentences")
    
    # Print chunk statistics
    if chunks:
        avg_words = np.mean([c['word_count'] for c in chunks])
        print(f"    Avg chunk size: {avg_words:.1f} words")
    
    return chunks


# =============================================
# Deduplication (Remove duplicate information)
# =============================================

def deduplicate_chunks(
    chunks: List[Dict[str, any]], 
    similarity_threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[List[Dict[str, any]], List[int]]:
    """
    Remove near-duplicate chunks using embedding-based cosine similarity.
    
    Process:
    1. Generate embeddings for all chunks
    2. For each chunk, compare with all previously kept chunks
    3. If max similarity > threshold, mark as duplicate
    4. Keep first occurrence, discard duplicates
    
    This is crucial for PV because:
    - Reduces storage and retrieval costs
    - Prevents redundant information in context
    - Improves retrieval relevance
    
    Args:
        chunks: List of chunk dictionaries
        similarity_threshold: Cosine similarity threshold (0-1)
                             0.85-0.90 recommended for near-duplicate detection
    
    Returns:
        Tuple of (unique_chunks, kept_original_indices)
    """
    if not chunks:
        return [], []
    
    chunk_texts = [c['text'] for c in chunks]
    
    print(f" Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embedding_model.encode(chunk_texts, show_progress_bar=False)
    embeddings_array = np.array(embeddings)
    
    unique_chunks = []
    unique_embeddings = []
    kept_indices = []
    duplicate_count = 0
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_array)):
        is_duplicate = False
        max_similarity = 0.0
        
        # Compare with all previously kept chunks
        if unique_embeddings:
            similarities = cosine_similarity(
                embedding.reshape(1, -1), 
                np.array(unique_embeddings)
            )[0]
            
            max_similarity = np.max(similarities)
            
            if max_similarity > similarity_threshold:
                is_duplicate = True
                duplicate_count += 1
                print(f"    Chunk {i} is {max_similarity:.2%} similar to existing chunk - skipping")
        
        # Keep chunk if not duplicate
        if not is_duplicate:
            unique_chunks.append(chunk)
            unique_embeddings.append(embedding)
            kept_indices.append(i)
    
    dedup_rate = (duplicate_count / len(chunks)) * 100 if chunks else 0
    print(f" Deduplication complete: Kept {len(unique_chunks)}/{len(chunks)} chunks "
          f"({duplicate_count} duplicates removed, {dedup_rate:.1f}% reduction)")
    
    return unique_chunks, kept_indices


# =========================
# Embeddings Generation
# =========================

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate semantic embeddings using sentence-transformers.
    
    Model: all-MiniLM-L6-v2
    - Dimension: 384
    - Fast inference
    - Good for semantic similarity
    """
    if not texts:
        return []
    
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


# =====================================
# Store PDF in ChromaDB 
# =====================================

def embed_and_store_pdf(
    pdf_text: str, 
    report_id: str,
    use_semantic_chunking: bool = True,
    apply_deduplication: bool = True,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Complete pipeline: Extract → Normalize → Chunk → Deduplicate → Embed → Store
    
    Args:
        pdf_text: Extracted PDF text
        report_id: Unique identifier for the adverse event report
        use_semantic_chunking: Use semantic vs fixed-size chunking
        apply_deduplication: Remove near-duplicates
        similarity_threshold: Threshold for deduplication
        verbose: Print detailed progress
    
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        'report_id': report_id,
        'original_text_length': len(pdf_text),
        'total_chunks_created': 0,
        'chunks_after_dedup': 0,
        'duplicates_removed': 0,
        'chunks_stored': 0
    }
    
    # Step 1: Normalize
    if verbose:
        print(f"\n{'='*60}")
        print(f" Processing Report: {report_id}")
        print(f"{'='*60}")
    
    normalized_text = normalize_text(pdf_text)
    
    if not normalized_text:
        print(" No text after normalization")
        return stats
    
    # Step 2: Semantic Chunking
    if use_semantic_chunking:
        if verbose:
            print("\n Step 1: Semantic Chunking")
        chunks = semantic_chunk_text(
            normalized_text,
            max_chunk_size=MAX_CHUNK_SIZE,
            min_chunk_size=MIN_CHUNK_SIZE,
            overlap_sentences=OVERLAP_SENTENCES
        )
    else:
        if verbose:
            print("\n Step 1: Fixed-Size Chunking (fallback)")
        # Simple fixed-size chunking
        words = normalized_text.split()
        chunk_size = 500
        overlap = 50
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'start_sentence': i // chunk_size,
                    'end_sentence': (i + chunk_size) // chunk_size,
                    'word_count': len(chunk_text.split())
                })
    
    stats['total_chunks_created'] = len(chunks)
    
    if not chunks:
        print(" No chunks created")
        return stats
    
    # Step 3: Deduplication
    kept_indices = list(range(len(chunks)))
    
    if apply_deduplication and len(chunks) > 1:
        if verbose:
            print(f"\n🔍 Step 2: Deduplication (threshold={similarity_threshold})")
        chunks, kept_indices = deduplicate_chunks(chunks, similarity_threshold)
        stats['chunks_after_dedup'] = len(chunks)
        stats['duplicates_removed'] = stats['total_chunks_created'] - len(chunks)
    else:
        stats['chunks_after_dedup'] = len(chunks)
    
    # Step 4: Generate Embeddings
    if verbose:
        print(f"\n Step 3: Generating Embeddings")
    
    chunk_texts = [c['text'] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    
    # Step 5: Store in ChromaDB
    if verbose:
        print(f"\n Step 4: Storing in Vector Database")
    
    collection = get_or_create_collection()
    
    # Generate IDs
    chunk_ids = [f"{report_id}_chunk_{idx}" for idx in kept_indices]
    
    # Metadata
    metadatas = [
        {
            "report_id": report_id,
            "chunk_index": idx,
            "original_chunk_number": idx,
            "start_sentence": chunk.get('start_sentence', 0),
            "end_sentence": chunk.get('end_sentence', 0),
            "word_count": chunk.get('word_count', 0),
            "semantic_chunking": use_semantic_chunking,
            "deduplication_applied": apply_deduplication,
            "similarity_threshold": similarity_threshold
        }
        for idx, chunk in zip(kept_indices, chunks)
    ]
    
    # Store
    try:
        collection.add(
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas,
            ids=chunk_ids
        )
        stats['chunks_stored'] = len(chunks)
        
        if verbose:
            print(f" Successfully stored {len(chunks)} chunks for report {report_id}")
    except Exception as e:
        print(f" Storage error: {e}")
        stats['error'] = str(e)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f" Processing Summary:")
        print(f"   • Original text: {stats['original_text_length']} characters")
        print(f"   • Chunks created: {stats['total_chunks_created']}")
        print(f"   • After deduplication: {stats['chunks_after_dedup']}")
        print(f"   • Duplicates removed: {stats['duplicates_removed']}")
        print(f"   • Chunks stored: {stats['chunks_stored']}")
        print(f"{'='*60}\n")
    
    return stats


# ===================================================================
# RAG Retrieval with Enhanced Context
# Find relevant information from a specific report
# ===================================================================

def retrieve_context(
    query: str, 
    report_id: str, 
    top_k: int = 5,
    verbose: bool = True
) -> Tuple[List[str], List[Dict]]:
    """
    Retrieve most relevant chunks for a specific report.
    
    Critical for pharmacovigilance:
    - Report-level isolation prevents cross-contamination
    - Semantic search finds relevant context even with different wording
    - Metadata provides traceability
    
    Args:
        query: Search query (e.g., "What adverse events occurred?")
        report_id: Report to search within
        top_k: Number of chunks to retrieve
        verbose: Print retrieval details
    
    Returns:
        Tuple of (documents, metadatas)
    """
    collection = get_or_create_collection()
    
    # Normalize query
    normalized_query = normalize_text(query)
    
    if verbose:
        print(f"\n🔍 Retrieving context for: '{query}'")
        print(f"   Report ID: {report_id}")
        print(f"   Top-K: {top_k}")
    
    # Generate query embedding
    query_embedding = generate_embeddings([normalized_query])[0]
    
    # Query ChromaDB with report_id filter
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"report_id": report_id}
        )
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        if verbose:
            print(f" Retrieved {len(documents)} chunks")
            
            if documents:
                print("\n Retrieved Chunks:")
                for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
                    similarity = 1 - dist  # Convert distance to similarity
                    print(f"\n   [{i}] Similarity: {similarity:.2%}")
                    print(f"       Chunk {meta.get('chunk_index', '?')}: {doc[:150]}...")
        
        return documents, metadatas
        
    except Exception as e:
        print(f" Retrieval error: {e}")
        return [], []


# ================================
# Analytics & Utilities
# ================================

def get_report_stats(report_id: str) -> Dict[str, any]:
    """Get detailed statistics for a stored report."""
    collection = get_or_create_collection()
    
    try:
        results = collection.get(
            where={"report_id": report_id}
        )
        
        metadatas = results.get("metadatas", [])
        
        if not metadatas:
            return {"error": f"No chunks found for report {report_id}"}
        
        total_chunks = len(metadatas)
        dedup_applied = metadatas[0].get("deduplication_applied", False)
        semantic = metadatas[0].get("semantic_chunking", False)
        
        word_counts = [m.get("word_count", 0) for m in metadatas]
        
        return {
            "report_id": report_id,
            "total_chunks": total_chunks,
            "semantic_chunking": semantic,
            "deduplication_applied": dedup_applied,
            "avg_chunk_size": np.mean(word_counts) if word_counts else 0,
            "min_chunk_size": min(word_counts) if word_counts else 0,
            "max_chunk_size": max(word_counts) if word_counts else 0
        }
    except Exception as e:
        return {"error": str(e)}


def list_all_reports() -> List[str]:
    """List all report IDs in the database."""
    collection = get_or_create_collection()
    
    try:
        all_data = collection.get()
        metadatas = all_data.get("metadatas", [])
        
        report_ids = list(set(m.get("report_id") for m in metadatas if m.get("report_id")))
        
        print(f" Found {len(report_ids)} unique reports")
        return sorted(report_ids)
    except Exception as e:
        print(f" Error listing reports: {e}")
        return []


def clear_pdf_collection():
    """Clear all stored embeddings."""
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("  Collection cleared successfully")
    except Exception as e:
        print(f"  Failed to clear collection: {e}")

