"""
One-time backfill script to index historical audit_log records
into the vector database for semantic similarity.

IMPORTANT:
Uses REAL database column names (snake_case),
NOT UI/export display names.
"""

from app.database.db import get_connection
from app.services.vector_service import index_audit_event


def backfill():
    print("[Backfill] Starting audit_log backfill into vector DB...")

    with get_connection() as conn:
        cursor = conn.cursor()

        # DEBUG: confirm row count first
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        total = cursor.fetchone()[0]
        print(f"[Backfill][DEBUG] Total rows in audit_log: {total}")

        # REAL query using REAL DB column names
        cursor.execute("""
            SELECT 
                report_id,
                drugname,
                adverse_event,
                risk_level,
                escalation_decision,
                timestamp
            FROM audit_log
            WHERE escalation_decision = 'ESCALATE'
        """)

        rows = cursor.fetchall()
        print(f"[Backfill] Found {len(rows)} historical escalated records")

        for row in rows:
            record = dict(row)

            success = index_audit_event(
                report_id=record["report_id"],
                drugname=record["drugname"],
                adverse_event=record["adverse_event"],
                risk_level=record["risk_level"],
                escalation_decision=record["escalation_decision"],
                symptoms=[],                 # not stored in DB
                ml_probability=0.0,          # not required for backfill
                timestamp=record.get("timestamp")
            )

            if success:
                print(f"[Backfill] ✓ Indexed {record['report_id']}")
            else:
                print(f"[Backfill] ✗ Failed {record['report_id']}")

    print("[Backfill] Completed backfill process.")


if __name__ == "__main__":
    backfill()
