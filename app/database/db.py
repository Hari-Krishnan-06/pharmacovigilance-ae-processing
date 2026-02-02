"""
SQLite Database Connection
Manages database for audit logging.
"""

import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator, Dict

#-----------------------------------------------------------
# ---------- DATABASE PATH ----------
#-----------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "audit.db"


#-----------------------------------------------------------
# ---------- INITIALIZATION ----------
#-----------------------------------------------------------

def init_database() -> None:
    """Initialize the database schema."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                drugname TEXT NOT NULL,
                adverse_event TEXT NOT NULL,
                ml_prediction TEXT,
                ml_probability REAL,
                extracted_drug TEXT,
                extracted_symptoms TEXT,
                escalation_decision TEXT,
                risk_level TEXT,
                final_score REAL,
                triggered_keywords TEXT,
                explanation TEXT,
                processing_time_ms REAL
            )
            """
        )

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_report_id ON audit_log(report_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_audit_risk_level ON audit_log(risk_level)"
        )

        conn.commit()

#-----------------------------------------------------------
# ---------- CONNECTION ----------
#-----------------------------------------------------------

@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=5.0,          # wait for locks instead of hanging
        check_same_thread=False
    )
    conn.row_factory = sqlite3.Row

    # ---- SQLite concurrency + performance pragmas ----
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")  # 5 seconds
    conn.execute("PRAGMA temp_store = MEMORY;")

    try:
        yield conn
    finally:
        conn.close()



# ---------- HEALTH / STATS ----------

def get_db_stats() -> Dict:
    try:
        with get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM audit_log")
            total = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM audit_log WHERE escalation_decision = 'ESCALATE'"
            )
            escalated = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM audit_log WHERE risk_level = 'CRITICAL'"
            )
            critical = cursor.fetchone()[0]

            return {
                "connected": True,
                "total_records": total,
                "escalated": escalated,
                "critical": critical,
                "database_path": str(DB_PATH),
            }

    except Exception:
        return {
            "connected": False,
            "total_records": 0,
            "escalated": 0,
            "critical": 0,
            "database_path": str(DB_PATH),
        }
