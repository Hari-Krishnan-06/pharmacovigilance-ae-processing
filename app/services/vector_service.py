"""
Vector Service
Handles semantic similarity search for adverse event reports using embeddings.

ARCHITECTURE:
- SQL (audit_log) = compliance, audit trail, regulatory history
- Vector DB (ChromaDB) = semantic similarity, intelligent search
- This service bridges the two systems

WORKFLOW:
1. Every audit log → canonical embedding text
2. Embedding text → vector (via sentence-transformers)
3. Vector → ChromaDB index
4. Query → retrieve similar cases via semantic search
"""

import os
import json
from typing import List, Dict
from datetime import datetime

# ChromaDB for vector storage
import chromadb
from chromadb.config import Settings

# Sentence Transformers for embeddings
from sentence_transformers import SentenceTransformer

# ============================
# CONFIG
# ============================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "adverse_events"

# ============================
# SINGLETONS (LAZY INIT)
# ============================

_embedding_model = None
_chroma_client = None
_collection = None


# ============================
# INITIALIZATION HELPERS
# ============================

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[Vector] Loading embedding model: {MODEL_NAME}")
        _embedding_model = SentenceTransformer(MODEL_NAME)
        print("[Vector] Embedding model loaded")
    return _embedding_model


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        print(f"[Vector] Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        print("[Vector] ChromaDB client initialized")
    return _chroma_client


def get_collection():
    global _collection
    if _collection is None:
        client = get_chroma_client()
        try:
            _collection = client.get_collection(name=COLLECTION_NAME)
            print(f"[Vector] Connected to existing collection: {COLLECTION_NAME}")
        except Exception:
            _collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"description": "Pharmacovigilance adverse event reports"}
            )
            print(f"[Vector] Created new collection: {COLLECTION_NAME}")
    return _collection


# ============================
# EMBEDDING TEXT
# ============================

def build_canonical_embedding_text(
    drugname: str,
    adverse_event: str,
    risk_level: str,
    escalation_decision: str,
    symptoms: List[str] = None
) -> str:
    symptoms_text = ", ".join(symptoms) if symptoms else "None extracted"

    return f"""Drug: {drugname}
Event: {adverse_event}
Symptoms: {symptoms_text}
Risk: {risk_level}
Decision: {escalation_decision}"""


def create_embedding(text: str) -> List[float]:
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


# ============================
# INDEXING
# ============================

def index_audit_event(
    report_id: str,
    drugname: str,
    adverse_event: str,
    risk_level: str,
    escalation_decision: str,
    symptoms: List[str] = None,
    ml_probability: float = 0.0,
    timestamp: str = None
) -> bool:
    """
    Index or update a single audit event in the vector database.

    MUST be called AFTER SQL commit.
    Uses UPSERT to avoid duplicate/index errors.
    """
    try:
        embedding_text = build_canonical_embedding_text(
            drugname=drugname,
            adverse_event=adverse_event,
            risk_level=risk_level,
            escalation_decision=escalation_decision,
            symptoms=symptoms or []
        )

        embedding = create_embedding(embedding_text)

        metadata = {
            "report_id": report_id,
            "drugname": drugname,
            "risk_level": risk_level,
            "escalation_decision": escalation_decision,
            "ml_probability": ml_probability,
            "timestamp": timestamp or datetime.utcnow().isoformat(),
            "symptoms": json.dumps(symptoms or [])
        }

        collection = get_collection()

        # CRITICAL: use UPSERT (not add)
        collection.upsert(
            ids=[report_id],
            embeddings=[embedding],
            documents=[embedding_text],
            metadatas=[metadata]
        )

        print(f"[Vector] Indexed event: {report_id} ({drugname})")
        return True

    except Exception as e:
        print(f"[Vector][ERROR] Failed to index {report_id}: {e}")
        return False


# ============================
# SEARCH
# ============================

def search_similar_events(
    query_drugname: str,
    query_event: str,
    query_symptoms: List[str],
    query_risk_level: str = None,
    current_report_id: str = None,
    top_k: int = 5,
    escalated_only: bool = True
) -> List[Dict]:
    """
    Semantic similarity search using vector DB.
    REPLACES SQL keyword-based similarity.
    """
    try:
        query_text = build_canonical_embedding_text(
            drugname=query_drugname,
            adverse_event=query_event,
            risk_level=query_risk_level or "UNKNOWN",
            escalation_decision="QUERY",
            symptoms=query_symptoms
        )

        query_embedding = create_embedding(query_text)

        where_filter = {}
        if escalated_only:
            where_filter["escalation_decision"] = "ESCALATE"

        collection = get_collection()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k + 10,  # get extra for filtering
            where=where_filter if where_filter else None
        )

        similar_events = []

        if results and results.get("ids") and results["ids"][0]:
            for i, report_id in enumerate(results["ids"][0]):
                if current_report_id and report_id == current_report_id:
                    continue

                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                document = results["documents"][0][i]

                try:
                    symptoms = json.loads(metadata.get("symptoms", "[]"))
                except Exception:
                    symptoms = []

                matched_symptoms = [
                    s for s in symptoms
                    if s.lower() in [q.lower() for q in (query_symptoms or [])]
                ]

                similar_events.append({
                    "report_id": report_id,
                    "drugname": metadata.get("drugname", ""),
                    "risk_level": metadata.get("risk_level", "UNKNOWN"),
                    "timestamp": metadata.get("timestamp", ""),
                    "ml_probability": metadata.get("ml_probability", 0.0),
                    "all_symptoms": symptoms,
                    "matched_symptoms": matched_symptoms,

                    # Honest metrics
                    "vector_distance": distance,

                    # For UI / explainability
                    "canonical_text": document
                })

                if len(similar_events) >= top_k:
                    break

        print(f"[Vector] Found {len(similar_events)} similar events for {query_drugname}")
        return similar_events

    except Exception as e:
        print(f"[Vector][ERROR] Vector search failed: {e}")
        return []


# ============================
# STATS / MAINTENANCE
# ============================

def get_collection_stats() -> Dict:
    try:
        collection = get_collection()
        return {
            "total_indexed": collection.count(),
            "collection_name": COLLECTION_NAME,
            "model": MODEL_NAME,
            "embedding_dimensions": 384
        }
    except Exception as e:
        return {
            "error": str(e),
            "total_indexed": 0
        }


def reset_collection() -> bool:
    """
    DANGER: Deletes all vectors. Use only in dev/testing.
    """
    try:
        client = get_chroma_client()
        client.delete_collection(name=COLLECTION_NAME)

        global _collection
        _collection = None

        get_collection()

        print(f"[Vector] Collection {COLLECTION_NAME} reset")
        return True

    except Exception as e:
        print(f"[Vector][ERROR] Failed to reset collection: {e}")
        return False
