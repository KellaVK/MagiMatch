"""
processor.py — Step 3: Loads all scraped JSON batches into SQLite.

Usage:
    PYTHONPATH=. python src/processor.py           # Normal run
    PYTHONPATH=. python src/processor.py --resume  # Resume after interruption
    PYTHONPATH=. python src/processor.py --reset   # Wipe DB and start over
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

from src.checkpoint import Checkpoint
from src.config_loader import cfg, paths
from src.db import get_connection, init_schema
from src.logger import get_logger

logger = get_logger("processor")

# ── Filters ────────────────────────────────────────────────────────────────────

FRONT_MATTER_TITLES = {t.lower() for t in cfg.processor.front_matter_titles}

JUNK_CATEGORIES = {c.lower() for c in cfg.processor.junk_categories}


def _is_front_matter(title: str) -> bool:
    if not title:
        return False
    t = title.strip().lower()
    # Exact match
    if t in FRONT_MATTER_TITLES:
        return True
    # Starts-with match (e.g. "Introduction to the Second Edition")
    for fm in FRONT_MATTER_TITLES:
        if t.startswith(fm):
            return True
    return False


def _clean_category(cat: str) -> str | None:
    if not cat:
        return None
    c = cat.strip()
    if c.lower() in JUNK_CATEGORIES:
        return None
    return c


# ── Person helpers ──────────────────────────────────────────────────────────────

def _canonical(name: str) -> str:
    """Normalize a person name for deduplication."""
    return re.sub(r"\s+", " ", name.strip()).title()


def _upsert_person(conn, name_cache: dict, name: str) -> int:
    """Return person_id, creating the person if new."""
    canon = _canonical(name)
    if canon in name_cache:
        return name_cache[canon]
    row = conn.execute("SELECT id FROM persons WHERE canonical_name = ?", (canon,)).fetchone()
    if row:
        pid = row["id"]
    else:
        cur = conn.execute("INSERT INTO persons (canonical_name, roles) VALUES (?, ?)", (canon, ""))
        pid = cur.lastrowid
    name_cache[canon] = pid
    return pid


# ── embed_text builder ─────────────────────────────────────────────────────────

def _build_embed_text(trick_title: str, description: str, book_title: str, authors: list, categories: list) -> str:
    author_str = ", ".join(authors) if authors else "Unknown"
    cat_str = ", ".join(categories) if categories else ""
    desc = description.strip() if description else ""
    parts = [f"{trick_title}."]
    if desc:
        parts.append(desc)
    parts.append(f"From: {book_title} by {author_str}.")
    if cat_str:
        parts.append(f"[{cat_str}]")
    return " ".join(parts)


# ── Main loader ────────────────────────────────────────────────────────────────

def load_batch_file(filepath: Path, conn, name_cache: dict, checkpoint: Checkpoint):
    batch_key = filepath.name
    if checkpoint.is_done(batch_key):
        logger.debug(f"Skipping already-processed batch: {batch_key}")
        return 0

    with open(filepath, "r", encoding="utf-8") as f:
        books = json.load(f)

    books_loaded = 0

    for book_data in books:
        archive_id = book_data.get("archive_id")
        if archive_id is None:
            continue

        title = (book_data.get("title") or {}).get("value") or "Unknown Title"
        pub_year = (book_data.get("pub_year") or {}).get("value")
        publisher = (book_data.get("publisher") or {}).get("value")
        page_count_raw = (book_data.get("page_count") or {}).get("value")
        page_count = int(page_count_raw) if page_count_raw and str(page_count_raw).isdigit() else None
        language = (book_data.get("language") or {}).get("value", "English")
        entry_count = book_data.get("entry_count")

        # Upsert book
        existing = conn.execute("SELECT id FROM books WHERE archive_id = ?", (archive_id,)).fetchone()
        if existing:
            book_id = existing["id"]
        else:
            cur = conn.execute(
                """INSERT INTO books (archive_id, title, pub_year, publisher, page_count, language, entry_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (archive_id, title, pub_year, publisher, page_count, language, entry_count),
            )
            book_id = cur.lastrowid

        # Authors
        author_names = []
        for person in book_data.get("authors", []):
            name = person.get("name", "").strip()
            if not name:
                continue
            pid = _upsert_person(conn, name_cache, name)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO book_persons (book_id, person_id, role) VALUES (?, ?, 'author')",
                    (book_id, pid),
                )
            except Exception:
                pass
            author_names.append(_canonical(name))

        # Subjects
        for person in book_data.get("subjects", []):
            name = person.get("name", "").strip()
            if not name:
                continue
            pid = _upsert_person(conn, name_cache, name)
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO book_persons (book_id, person_id, role) VALUES (?, ?, 'subject')",
                    (book_id, pid),
                )
            except Exception:
                pass

        # Tricks
        for trick in book_data.get("tricks", []):
            t_title = (trick.get("title") or {}).get("value", "").strip()
            if not t_title or _is_front_matter(t_title):
                continue

            raw_cats = trick.get("categories", [])
            categories = [c for c in (raw_cats or []) if _clean_category(c)]
            if not categories and raw_cats:
                # All categories were junk — skip trick
                continue

            description = (trick.get("description") or {}).get("value", "")
            page_number = (trick.get("page_number") or {}).get("value")
            archive_entry_id = trick.get("archive_entry_id")

            # Primary category
            effect_category = categories[0] if categories else None

            # Credited to (first creator)
            creators = trick.get("creators") or []
            credited_to = None
            inventor_id = None
            if creators:
                first_creator = creators[0].get("name", "").strip()
                if first_creator:
                    credited_to = _canonical(first_creator)
                    inventor_id = _upsert_person(conn, name_cache, first_creator)

            embed_text = _build_embed_text(t_title, description, title, author_names, categories)

            conn.execute(
                """INSERT INTO tricks
                   (book_id, archive_entry_id, title, page_number, description,
                    effect_category, credited_to, inventor_id, embed_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (book_id, archive_entry_id, t_title, page_number, description,
                 effect_category, credited_to, inventor_id, embed_text),
            )

        books_loaded += 1

    conn.commit()
    checkpoint.mark_done(batch_key)
    return books_loaded


def build_book_relations(conn):
    """Build book_relations table: same_author and shared_inventor links."""
    logger.info("Building book_relations (same_author)...")
    conn.execute("DELETE FROM book_relations WHERE relation_type = 'same_author'")
    conn.execute("""
        INSERT OR IGNORE INTO book_relations (book_a_id, book_b_id, relation_type, weight)
        SELECT DISTINCT bp1.book_id, bp2.book_id, 'same_author', 1.0
        FROM book_persons bp1
        JOIN book_persons bp2 ON bp1.person_id = bp2.person_id
            AND bp1.book_id < bp2.book_id
            AND bp1.role = 'author'
            AND bp2.role = 'author'
    """)

    logger.info("Building book_relations (shared_inventor)...")
    conn.execute("DELETE FROM book_relations WHERE relation_type = 'shared_inventor'")
    conn.execute("""
        INSERT OR IGNORE INTO book_relations (book_a_id, book_b_id, relation_type, weight)
        SELECT DISTINCT t1.book_id, t2.book_id, 'shared_inventor', 1.0
        FROM tricks t1
        JOIN tricks t2 ON t1.inventor_id = t2.inventor_id
            AND t1.book_id < t2.book_id
            AND t1.inventor_id IS NOT NULL
    """)
    conn.commit()
    logger.info("book_relations built.")


def run(resume: bool = False, reset: bool = False):
    db_path = paths.db
    checkpoint_dir = paths.checkpoints

    if reset:
        logger.info("Reset flag set — wiping database and checkpoint.")
        if db_path.exists():
            db_path.unlink()
        Checkpoint("processor", checkpoint_dir).reset()

    init_schema(db_path)
    conn = get_connection(db_path)
    checkpoint = Checkpoint("processor", checkpoint_dir)
    name_cache: dict = {}

    # Pre-populate name cache from existing persons
    for row in conn.execute("SELECT id, canonical_name FROM persons"):
        name_cache[row["canonical_name"]] = row["id"]

    raw_dir = paths.raw_data
    batch_files = sorted(Path(raw_dir).glob("books_*_batch*.json"))
    logger.info(f"Found {len(batch_files)} batch files in {raw_dir}")

    total_books = 0
    for i, bf in enumerate(batch_files):
        loaded = load_batch_file(bf, conn, name_cache, checkpoint)
        total_books += loaded
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(batch_files)} batches processed ({total_books} books this run)")

    logger.info(f"All batches loaded. Building relations...")
    build_book_relations(conn)

    # Final counts
    book_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    trick_count = conn.execute("SELECT COUNT(*) FROM tricks").fetchone()[0]
    person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    logger.info(f"Done. DB contains: {book_count} books, {trick_count} tricks, {person_count} persons.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MagiMatch Step 3 — JSON → SQLite processor")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--reset", action="store_true", help="Wipe DB and start fresh")
    args = parser.parse_args()
    run(resume=args.resume, reset=args.reset)
