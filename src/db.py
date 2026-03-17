"""
db.py — SQLite connection and schema initialization for MagiMatch.
"""

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS books (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_id      INTEGER UNIQUE NOT NULL,
    title           TEXT NOT NULL,
    pub_year        TEXT,
    publisher       TEXT,
    page_count      INTEGER,
    language        TEXT DEFAULT 'English',
    entry_count     INTEGER,
    obscurity_score REAL DEFAULT 0.5,
    data_source     TEXT DEFAULT 'scraped'
);

CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT UNIQUE NOT NULL,
    roles           TEXT    -- comma-separated: 'author', 'subject', 'inventor'
);

CREATE TABLE IF NOT EXISTS book_persons (
    book_id         INTEGER NOT NULL REFERENCES books(id),
    person_id       INTEGER NOT NULL REFERENCES persons(id),
    role            TEXT NOT NULL,   -- 'author' | 'subject'
    PRIMARY KEY (book_id, person_id, role)
);

CREATE TABLE IF NOT EXISTS tricks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id         INTEGER NOT NULL REFERENCES books(id),
    archive_entry_id TEXT,
    title           TEXT NOT NULL,
    page_number     TEXT,
    description     TEXT,
    effect_category TEXT,
    credited_to     TEXT,
    inventor_id     INTEGER REFERENCES persons(id),
    embed_text      TEXT,
    data_source     TEXT DEFAULT 'scraped'
);

CREATE TABLE IF NOT EXISTS book_relations (
    book_a_id       INTEGER NOT NULL REFERENCES books(id),
    book_b_id       INTEGER NOT NULL REFERENCES books(id),
    relation_type   TEXT NOT NULL,  -- 'same_author' | 'shared_inventor' | 'prop_overlap'
    weight          REAL DEFAULT 1.0,
    PRIMARY KEY (book_a_id, book_b_id, relation_type)
);

CREATE TABLE IF NOT EXISTS enrichment_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT,
    entity_id       INTEGER,
    field_name      TEXT,
    filled_value    TEXT,
    prompt_used     TEXT,
    model_used      TEXT,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embedding_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type     TEXT NOT NULL,
    model_name      TEXT NOT NULL,
    record_count    INTEGER,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tricks_book_id ON tricks(book_id);
CREATE INDEX IF NOT EXISTS idx_tricks_inventor_id ON tricks(inventor_id);
CREATE INDEX IF NOT EXISTS idx_tricks_effect_category ON tricks(effect_category);
CREATE INDEX IF NOT EXISTS idx_book_persons_book_id ON book_persons(book_id);
CREATE INDEX IF NOT EXISTS idx_book_persons_person_id ON book_persons(person_id);
CREATE INDEX IF NOT EXISTS idx_books_archive_id ON books(archive_id);
CREATE INDEX IF NOT EXISTS idx_persons_canonical_name ON persons(canonical_name);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """
    Open a SQLite connection with WAL mode and check_same_thread=False
    (required for Gradio's multi-threaded environment).
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(db_path: Path):
    """Create all tables if they don't exist."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()
