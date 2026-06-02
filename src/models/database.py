import sqlite3
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("SCHOLAI_DB", "data/scholai.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


#  connection helper 
@contextmanager
def get_db():
    """Yield a connected, row-factory–enabled SQLite connection."""
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row         
    conn.execute("PRAGMA journal_mode=WAL") 
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


#  schema creation 
def init_db():
    """Create all tables if they don't already exist."""
    _ensure_dir()
    with get_db() as conn:
        conn.executescript("""
            -- ── users ──────────────────────────────────────────
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- ── upload_history ──────────────────────────────────
            CREATE TABLE IF NOT EXISTS upload_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename    TEXT    NOT NULL,
                file_size   INTEGER,          -- bytes
                chunks      INTEGER,          -- vector chunks produced
                uploaded_at TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- ── query_history ───────────────────────────────────
            CREATE TABLE IF NOT EXISTS query_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                question  TEXT    NOT NULL,
                answer    TEXT    NOT NULL,
                marks     INTEGER NOT NULL DEFAULT 10,
                language  TEXT    NOT NULL DEFAULT 'English',
                asked_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            -- indices for fast per-user queries
            CREATE INDEX IF NOT EXISTS idx_upload_user ON upload_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_query_user  ON query_history(user_id);
        """)
    print(f"[DB] Initialised → {DB_PATH}")


#  tiny query helpers (used by auth.py & app.py) 
def fetch_one(sql: str, params: tuple = ()):
    with get_db() as conn:
        return conn.execute(sql, params).fetchone()

def fetch_all(sql: str, params: tuple = ()):
    with get_db() as conn:
        return conn.execute(sql, params).fetchall()

def execute(sql: str, params: tuple = ()):
    """Run INSERT / UPDATE / DELETE and return lastrowid."""
    with get_db() as conn:
        cur = conn.execute(sql, params)
        return cur.lastrowid