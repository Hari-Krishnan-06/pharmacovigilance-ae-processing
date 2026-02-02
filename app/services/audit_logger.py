"""
Audit Logger Service
Logs all processing decisions for regulatory compliance.

ARCHITECTURE:
- SQL (audit_log) = compliance, audit trail, regulatory history
- Vector DB = semantic similarity for related adverse events
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.database.db import get_connection

# ============================
# VECTOR SERVICE (NEW)
# ============================

from app.services.vector_service import (
    index_audit_event,
    search_similar_events
)


# ============================
# CORE LOGGING
# ============================

def generate_report_id() -> str:
    """Generate a unique report ID."""
    return f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"


def log_processing(
    report_id: str,
    drugname: str,
    adverse_event: str,
    ml_prediction: Dict,
    entities: Dict,
    escalation_result,
    processing_time_ms: float
) -> bool:
    """
    Log a processing decision to the audit database
    AND index it into the vector database.
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO audit_log (
                    report_id, drugname, adverse_event,
                    ml_prediction, ml_probability,
                    extracted_drug, extracted_symptoms,
                    escalation_decision, risk_level, final_score,
                    triggered_keywords, explanation, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                report_id,
                drugname,
                adverse_event,
                ml_prediction.get("prediction", ""),
                ml_prediction.get("serious_probability", 0),
                entities.get("drug", ""),
                json.dumps(entities.get("symptoms", [])),
                "ESCALATE" if escalation_result.should_escalate else "NO_ESCALATE",
                escalation_result.risk_level,
                escalation_result.final_score,
                json.dumps(escalation_result.triggered_keywords),
                escalation_result.explanation,
                processing_time_ms
            ))

            conn.commit()

        # ============================
        # VECTOR INDEXING (NEW)
        # ============================
        try:
            index_audit_event(
                report_id=report_id,
                drugname=drugname,
                adverse_event=adverse_event,
                risk_level=escalation_result.risk_level,
                escalation_decision="ESCALATE" if escalation_result.should_escalate else "NO_ESCALATE",
                symptoms=entities.get("symptoms", []),
                ml_probability=ml_prediction.get("serious_probability", 0),
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as ve:
            # Do NOT break audit flow if vector fails
            print(f"[WARN] Vector indexing failed for {report_id}: {ve}")

        return True

    except Exception as e:
        print(f"Error logging to audit: {e}")
        return False


# ============================
# STANDARD AUDIT RETRIEVAL
# ============================

def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    risk_level: Optional[str] = None,
    escalated_only: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict]:
    """Retrieve audit logs with optional filtering."""
    with get_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM audit_log WHERE 1=1"
        params = []

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)

        if escalated_only:
            query += " AND escalation_decision = 'ESCALATE'"

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date + " 23:59:59")

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]


def get_audit_by_report_id(report_id: str) -> Optional[Dict]:
    """Get a specific audit record by report ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_log WHERE report_id = ?", (report_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_audit_summary() -> Dict:
    """Get summary statistics from audit log."""
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM audit_log")
        total = cursor.fetchone()[0]

        cursor.execute("""
            SELECT risk_level, COUNT(*) as count 
            FROM audit_log 
            GROUP BY risk_level
        """)
        by_risk_level = {row['risk_level']: row['count'] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT 
                SUM(CASE WHEN escalation_decision = 'ESCALATE' THEN 1 ELSE 0 END) as escalated,
                COUNT(*) as total
            FROM audit_log
        """)
        row = cursor.fetchone()
        escalation_rate = row['escalated'] / row['total'] if row['total'] > 0 else 0

        cursor.execute("SELECT AVG(processing_time_ms) FROM audit_log")
        avg_processing_time = cursor.fetchone()[0] or 0

        return {
            "total_processed": total,
            "by_risk_level": by_risk_level,
            "escalation_rate": escalation_rate,
            "avg_processing_time_ms": avg_processing_time
        }


# ============================================================
# VECTOR-BASED SIMILARITY (NEW PRIMARY)
# ============================================================
def get_similar_serious_events_vector(
    drugname: str,
    adverse_event: str,
    current_symptoms: List[str],
    risk_level: str,
    current_report_id: str,
    limit: int = 5
) -> List[Dict]:
    try:
        raw_results = search_similar_events(
            query_drugname=drugname,
            query_event=adverse_event,
            query_symptoms=current_symptoms,
            query_risk_level=risk_level,
            current_report_id=current_report_id,
            top_k=limit,
            escalated_only=True
        )

        similar_events = []

        for row in raw_results:
            similar_events.append({
                "report_id": row.get("report_id"),
                "drugname": row.get("drugname"),

                # ✅ REAL FIX
                "adverse_event": row.get("canonical_text", ""),

                "timestamp": row.get("timestamp"),
                "risk_level": row.get("risk_level"),
                "ml_probability": row.get("ml_probability"),
                "final_score": None,  # not returned by vector
                "matched_symptoms": row.get("matched_symptoms", [])
            })

        print(f"[Vector] Returning {len(similar_events)} similar events for {drugname}")
        return similar_events

    except Exception as e:
        print(f"[ERROR] Vector similarity failed: {e}")
        return []


# ============================================================
# LEGACY SQL SIMILARITY (DO NOT USE AS PRIMARY)
# ============================================================

def get_similar_serious_events_drug_based(
    drugname: str,
    current_symptoms: List[str],
    current_report_id: str,
    limit: int = 5
) -> List[Dict]:
    """
    LEGACY: Drug-based SQL similarity.
    Kept only for debugging / fallback.
    """
    return []


def get_similar_serious_events(
    current_symptoms: List[str],
    current_report_id: str,
    limit: int = 5
) -> List[Dict]:
    """
    LEGACY: SQL keyword overlap similarity.
    DO NOT USE as primary anymore.
    """
    return []
