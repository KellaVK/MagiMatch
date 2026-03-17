"""
query_engine.py — MagiMatch v4 Query Engine.

Pipeline:
  1. Intent parsing (gpt-4o-mini)
  2. Person style expansion (profiles dict → Serper → DB fallback)
  3. Title pre-pass (SQL exact + fuzzy)
  4. Semantic search (cosine similarity over tricks.npy)
  5. Merge + deduplicate
  6. AI descriptions (batched gpt-4o-mini)
  7. Commentary (gpt-4o-mini)
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config_loader import paths
from src.db import get_connection
from src.logger import get_logger
from src.openai_client import chat_completion, embed_single

logger = get_logger("query_engine")

# ── Magician profiles ──────────────────────────────────────────────────────────
# Focus on what makes each person DISTINCTIVE, not their genre.
# These drive semantic query expansion.

MAGICIAN_PROFILES = {
    # Mentalists
    "matt mello": "propless mentalism entirely without props no cards no objects pure psychological effects direct mind reading no setup invisible technique",
    "max maven": "dark theatrical mentalism psychological atmosphere sinister presentation verbal control suggestion bizarre occult symbolism",
    "luke jermay": "intimate one-on-one mentalism cold reading psychological subtlety human connection emotional depth genuine connection performers",
    "bob cassidy": "structured mentalism systems book tests billet work professional nightclub formal mentalism analytical methodical",
    "richard osterlind": "impromptu mentalism anywhere anytime no props no setup maximum freedom borrowed objects everyday items",
    "annemann": "practical mentalism raw classic card mysteries billet work bulletin board classic methods foundational mental magic",
    "corinda": "thirteen steps complete mentalism system comprehensive stage mentalism crystal gazing second sight telepathy code systems",
    "al koran": "polished elegant commercial mentalism professional smooth sophisticated stage presence borrowed ring watch prediction",
    "lee earle": "social mentalism relationship dynamics human behavior psychological influence personal reading one-ahead systems",

    # Card workers
    "dai vernon": "subtlety depth psychology naturalness classic sleight of hand slow deceptive elegant the professor cardician",
    "ed marlo": "technical card mechanics prolific inventor false shuffles false cuts controls large body of work underground",
    "darwin ortiz": "technical precision card technique clarity rigorous theory card college structured approach refined mechanics",
    "larry jennings": "knacky sleights unusual angles visual card magic deceptive at close range underground cult following",
    "juan tamariz": "theatrical comedy chaos theory spectator fairness magician in trouble personality-driven card magic Spanish school",
    "simon aronson": "memory work stack principle stay-stack logical construction mathematical rigor theoretical purity",
    "arturo de ascanio": "Spanish school theoretical purity conceptual elegance philosophical approach analysis of magic grammar of magic",
    "rene lavand": "one-handed card magic disability restriction minimalism Argentine slow motion visual impossible poetic",
    "lennart green": "chaos disorder messy shuffles unpredictable visual impossible angle-proof spectator shuffles green angle proof",
    "roberto giobbi": "card college systematic complete comprehensive beginner to advanced structured technical encyclopedic",
    "paul curry": "out of this world spectator does the magic hands off self-working logic puzzles card puzzles",
    "s.w. erdnase": "expert at the card table classic bible gambling moves second deal bottom deal angle mechanics technical foundation",
    "frank garcia": "super subtle card sleights commercial close-up strong visual flashy effects restaurant workers",
    "dani daortiz": "chaos magic impossible moments spectator shuffles multiple packets confusion as method Spanish contemporary",

    # Coin workers
    "david roth": "expert coin magic professional coins across retention vanish structured complete coin magic systems",
    "slydini": "misdirection philosophy natural movement timing lap work tissue paper cleanest hands visual impossibility",
    "han ping chien": "han ping chien move coins across table surface coin migration simple direct visual",

    # Platform and stage
    "tommy wonder": "perfection detail obsessive engineering every angle every timing polished platform professional Dutch precision",
    "paul harris": "astonishment reality pop visual shock bizarre comedy surreal impossible moments art of astonishment",
    "derek dingle": "commercial close-up strong visual restaurant bar workers professional sleek polished routines",
    "michael close": "workers performance theory real-world close-up professional thinking theory worker approach",
    "jim steinmeyer": "illusion design impossible objects theatrical stage grand illusion engineering geometry misdirection grand scale",

    # Other
    "harry lorayne": "memory system card memory memorized deck mnemonics retention recall champion memory systems",
    "charlie miller": "elegant close-up classical sleight minimalist sophisticated refined quiet underground collector",
}

# Style cache: populated at runtime
_style_cache: dict = {}

# ── Serper web lookup ─────────────────────────────────────────────────────────

def _serper_lookup(magician_name: str) -> Optional[str]:
    """Use Serper to find a magician's style online, return raw snippet text."""
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        return None
    try:
        import requests
        query = f"{magician_name} magician magic style performance specialty"
        r = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        snippets = []
        for item in data.get("organic", [])[:5]:
            if item.get("snippet"):
                snippets.append(item["snippet"])
        return " ".join(snippets) if snippets else None
    except Exception as e:
        logger.warning(f"Serper lookup failed for {magician_name}: {e}")
        return None


def _db_style_lookup(magician_name: str, conn) -> Optional[str]:
    """Fallback: find credited tricks in DB to infer style."""
    rows = conn.execute(
        """SELECT t.embed_text FROM tricks t
           WHERE LOWER(t.credited_to) LIKE ?
           LIMIT 30""",
        (f"%{magician_name.lower()}%",),
    ).fetchall()
    if not rows:
        return None
    texts = " | ".join(r["embed_text"] for r in rows if r["embed_text"])
    return texts[:2000] if texts else None


def get_style_description(name: str, conn) -> str:
    """Return a keyword-rich style description for the named magician."""
    key = name.strip().lower()
    if key in _style_cache:
        return _style_cache[key]

    # 1. Curated profile
    if key in MAGICIAN_PROFILES:
        result = MAGICIAN_PROFILES[key]
        _style_cache[key] = result
        return result

    # 2. Serper web lookup → GPT summary
    web_text = _serper_lookup(name)
    if web_text:
        prompt = (
            f"In 20-30 words, describe the DISTINCTIVE magic style of {name} — "
            f"what makes their approach unique, what props they use, what effects they specialize in. "
            f"Focus on technical keywords a magician would search for. "
            f"Source text: {web_text[:1500]}"
        )
        try:
            summary = chat_completion(prompt, max_tokens=80, temperature=0.2)
            _style_cache[key] = summary
            return summary
        except Exception as e:
            logger.warning(f"GPT style summary failed for {name}: {e}")

    # 3. DB fallback
    db_text = _db_style_lookup(name, conn)
    if db_text:
        prompt = (
            f"In 20-30 words, describe the DISTINCTIVE magic style of {name} "
            f"based on these trick descriptions. Focus on keywords: {db_text[:1000]}"
        )
        try:
            summary = chat_completion(prompt, max_tokens=80, temperature=0.2)
            _style_cache[key] = summary
            return summary
        except Exception:
            pass

    # 4. Just use the name
    return name


# ── Intent parsing ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    """
    Parse the user query into structured intent using gpt-4o-mini.
    Returns: {topic, keywords, excluded_persons, referenced_persons, is_title_query}
    """
    system = """You parse magic trick search queries into structured JSON.
Return ONLY valid JSON with these fields:
- topic: string, the core topic/theme being searched
- keywords: list of strings, important search keywords
- excluded_persons: list of strings, names the user wants EXCLUDED (e.g. "not Vernon")
- referenced_persons: list of strings, magicians whose style is referenced (e.g. "like Mello")
- is_title_query: boolean, true if user is searching for a specific book or trick title
"""
    prompt = f'Parse this magic search query: "{query}"'
    try:
        raw = chat_completion(prompt, system=system, max_tokens=300, temperature=0.1, json_mode=True)
        result = json.loads(raw)
        return {
            "topic": result.get("topic", query),
            "keywords": result.get("keywords", []),
            "excluded_persons": [p.strip().lower() for p in result.get("excluded_persons", [])],
            "referenced_persons": [p.strip() for p in result.get("referenced_persons", [])],
            "is_title_query": bool(result.get("is_title_query", False)),
        }
    except Exception as e:
        logger.warning(f"Intent parsing failed: {e}")
        return {
            "topic": query,
            "keywords": [],
            "excluded_persons": [],
            "referenced_persons": [],
            "is_title_query": False,
        }


# ── Title pre-pass ─────────────────────────────────────────────────────────────

def title_search(query: str, conn) -> list:
    """
    Exact then fuzzy SQL title match. Returns list of result dicts.
    """
    results = []

    # Exact match
    rows = conn.execute(
        """SELECT t.id, t.title, t.description, t.effect_category, t.credited_to, t.embed_text,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.canonical_name) as authors
           FROM tricks t
           JOIN books b ON t.book_id = b.id
           LEFT JOIN book_persons bp ON b.id = bp.book_id AND bp.role = 'author'
           LEFT JOIN persons p ON bp.person_id = p.id
           WHERE LOWER(b.title) = LOWER(?)
           GROUP BY t.id
           LIMIT 10""",
        (query,),
    ).fetchall()
    for r in rows:
        results.append({**dict(r), "match_type": "title_exact"})

    if results:
        return results

    # Fuzzy: word-by-word LIKE
    words = [w for w in re.split(r"\s+", query.strip()) if len(w) >= 3]
    if not words:
        return []

    like_clauses = " AND ".join(f"LOWER(b.title) LIKE ?" for w in words)
    params = [f"%{w.lower()}%" for w in words]

    rows = conn.execute(
        f"""SELECT t.id, t.title, t.description, t.effect_category, t.credited_to, t.embed_text,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.canonical_name) as authors
           FROM tricks t
           JOIN books b ON t.book_id = b.id
           LEFT JOIN book_persons bp ON b.id = bp.book_id AND bp.role = 'author'
           LEFT JOIN persons p ON bp.person_id = p.id
           WHERE {like_clauses}
           GROUP BY t.id
           LIMIT 10""",
        params,
    ).fetchall()
    for r in rows:
        results.append({**dict(r), "match_type": "title_fuzzy"})

    return results


# ── Semantic search ────────────────────────────────────────────────────────────

def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query_vec and each row of matrix."""
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-9
    return matrix @ query_vec / (norms * query_norm)


def semantic_search(query_text: str, top_k: int = 40) -> list:
    """
    Embed query, compute cosine similarity over tricks.npy, return top_k trick IDs with scores.
    Returns list of (trick_id, score).
    """
    emb_dir = paths.embeddings
    tricks_npy = emb_dir / "tricks.npy"
    ids_npy = emb_dir / "trick_ids.npy"

    if not tricks_npy.exists() or not ids_npy.exists():
        logger.warning("Embeddings not found — semantic search unavailable.")
        return []

    embeddings = np.load(str(tricks_npy))
    trick_ids = np.load(str(ids_npy))

    query_vec = embed_single(query_text)
    scores = cosine_similarity_matrix(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [(int(trick_ids[i]), float(scores[i])) for i in top_indices]


# ── Fetch trick details ────────────────────────────────────────────────────────

def fetch_tricks(trick_ids: list, conn) -> dict:
    """Fetch full trick+book+author rows for a list of trick IDs. Returns dict keyed by trick_id."""
    if not trick_ids:
        return {}
    placeholders = ",".join("?" * len(trick_ids))
    rows = conn.execute(
        f"""SELECT t.id, t.title, t.description, t.effect_category, t.credited_to, t.embed_text,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.canonical_name) as authors
           FROM tricks t
           JOIN books b ON t.book_id = b.id
           LEFT JOIN book_persons bp ON b.id = bp.book_id AND bp.role = 'author'
           LEFT JOIN persons p ON bp.person_id = p.id
           WHERE t.id IN ({placeholders})
           GROUP BY t.id""",
        trick_ids,
    ).fetchall()
    return {r["id"]: dict(r) for r in rows}


# ── Merge + deduplicate ────────────────────────────────────────────────────────

def merge_results(
    title_results: list,
    semantic_results: list,
    trick_details: dict,
    excluded_persons: list,
    max_per_book: int = 2,
    max_per_author: int = 3,
    total: int = 10,
) -> list:
    """
    Merge title and semantic results, deduplicate, apply per-book/author caps.
    Excluded persons are soft-penalized (pushed toward end) rather than hard-filtered.
    """
    seen_ids = set()
    book_counts: dict = {}
    author_counts: dict = {}
    final = []

    def _add(result_dict, match_type, score=0.0):
        tid = result_dict["id"]
        if tid in seen_ids:
            return
        book_id = result_dict["book_id"]
        authors_str = result_dict.get("authors") or ""

        if book_counts.get(book_id, 0) >= max_per_book:
            return
        for a in authors_str.split(","):
            a = a.strip()
            if a and author_counts.get(a, 0) >= max_per_author:
                return

        seen_ids.add(tid)
        book_counts[book_id] = book_counts.get(book_id, 0) + 1
        for a in authors_str.split(","):
            a = a.strip()
            if a:
                author_counts[a] = author_counts.get(a, 0) + 1

        final.append({**result_dict, "match_type": match_type, "score": score})

    # Title results first (higher priority)
    for r in title_results:
        if len(final) >= total:
            break
        _add(r, r.get("match_type", "title_fuzzy"), score=1.0)

    # Semantic results (sort by score, penalize excluded)
    excluded_lower = [e.lower() for e in excluded_persons]

    def _is_excluded(r):
        credited = (r.get("credited_to") or "").lower()
        authors = (r.get("authors") or "").lower()
        return any(ex in credited or ex in authors for ex in excluded_lower)

    non_excluded = [(tid, s, trick_details[tid]) for tid, s in semantic_results if tid in trick_details and not _is_excluded(trick_details[tid])]
    penalized = [(tid, s, trick_details[tid]) for tid, s in semantic_results if tid in trick_details and _is_excluded(trick_details[tid])]

    for tid, score, r in non_excluded:
        if len(final) >= total:
            break
        _add(r, "semantic", score)

    for tid, score, r in penalized:
        if len(final) >= total:
            break
        _add(r, "semantic", score * 0.3)

    return final[:total]


# ── AI descriptions ────────────────────────────────────────────────────────────

def generate_descriptions(results: list) -> list:
    """
    Generate AI descriptions for all result cards in a single batched GPT call.
    Replaces sparse archive notes with meaningful 1-2 sentence explanations.
    """
    if not results:
        return results

    items = []
    for i, r in enumerate(results):
        items.append(
            f"{i+1}. Title: {r['title']}\n"
            f"   Book: {r.get('book_title','')}\n"
            f"   Description: {r.get('description','') or '(none)'}\n"
            f"   Category: {r.get('effect_category','') or '(unknown)'}\n"
            f"   Credited to: {r.get('credited_to','') or '(unknown)'}"
        )

    prompt = (
        "For each magic trick below, write 1-2 sentences explaining what it IS and why a magician might want it. "
        "Be specific. If the description is vague (like 'key card'), explain what that technique involves. "
        "Return ONLY a JSON array of strings, one per trick, in order.\n\n"
        + "\n\n".join(items)
    )

    try:
        raw = chat_completion(prompt, max_tokens=1200, temperature=0.4, json_mode=True)
        # Parse — might be {"descriptions": [...]} or just [...]
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            descs = list(parsed.values())[0]
        else:
            descs = parsed
        for i, r in enumerate(results):
            if i < len(descs) and descs[i]:
                r["ai_description"] = str(descs[i])
            else:
                r["ai_description"] = r.get("description") or ""
    except Exception as e:
        logger.warning(f"AI descriptions failed: {e}")
        for r in results:
            r["ai_description"] = r.get("description") or ""

    return results


# ── Commentary ────────────────────────────────────────────────────────────────

def generate_commentary(query: str, results: list) -> str:
    """2-3 sentence commentary explaining why these results match the query."""
    if not results:
        return "No results found for this query."
    titles = [f"{r['title']} ({r.get('book_title','')})" for r in results[:5]]
    prompt = (
        f'A magician searched for: "{query}"\n\n'
        f"The top results include: {', '.join(titles)}\n\n"
        "In 2-3 sentences, explain what connects these results to the search query. "
        "Be specific about the magic techniques or themes involved. Don't just list the titles."
    )
    try:
        return chat_completion(prompt, max_tokens=200, temperature=0.5)
    except Exception as e:
        logger.warning(f"Commentary generation failed: {e}")
        return ""


# ── Main search function ───────────────────────────────────────────────────────

def search(query: str, conn=None) -> dict:
    """
    Full MagiMatch search pipeline.
    Returns: {results: list, commentary: str, query_info: dict}
    """
    if conn is None:
        conn = get_connection(paths.db)

    # 1. Parse intent
    intent = parse_intent(query)
    logger.info(f"Intent: {intent}")

    # 2. Expand style if referenced persons exist
    expanded_query = intent["topic"]
    for person in intent["referenced_persons"]:
        style = get_style_description(person, conn)
        expanded_query = f"{expanded_query} {style}"

    # 3. Title pre-pass (if likely title query)
    title_results = []
    if intent["is_title_query"] or len(query.split()) <= 4:
        title_results = title_search(query, conn)
        if title_results and intent["is_title_query"]:
            # Pure title query — generate descriptions and return immediately
            title_results = generate_descriptions(title_results)
            commentary = generate_commentary(query, title_results)
            return {
                "results": title_results[:10],
                "commentary": commentary,
                "query_info": intent,
            }

    # 4. Semantic search
    sem_results = semantic_search(expanded_query, top_k=40)

    # 5. Fetch details for semantic results
    sem_ids = [tid for tid, _ in sem_results]
    trick_details = fetch_tricks(sem_ids, conn)

    # 6. Merge
    merged = merge_results(
        title_results=title_results,
        semantic_results=sem_results,
        trick_details=trick_details,
        excluded_persons=intent["excluded_persons"],
        total=10,
    )

    # 7. AI descriptions
    merged = generate_descriptions(merged)

    # 8. Commentary
    commentary = generate_commentary(query, merged)

    return {
        "results": merged,
        "commentary": commentary,
        "query_info": intent,
    }


# ── Browse helpers ─────────────────────────────────────────────────────────────

def get_all_effects(conn) -> list:
    """Return sorted list of effect categories."""
    rows = conn.execute(
        "SELECT DISTINCT effect_category FROM tricks WHERE effect_category IS NOT NULL ORDER BY effect_category"
    ).fetchall()
    return [r["effect_category"] for r in rows]


def browse_by_effect(effect: str, conn, limit: int = 20) -> list:
    """Return tricks matching an effect category."""
    rows = conn.execute(
        """SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.canonical_name) as authors
           FROM tricks t
           JOIN books b ON t.book_id = b.id
           LEFT JOIN book_persons bp ON b.id = bp.book_id AND bp.role = 'author'
           LEFT JOIN persons p ON bp.person_id = p.id
           WHERE t.effect_category = ?
           GROUP BY t.id
           ORDER BY RANDOM()
           LIMIT ?""",
        (effect, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_all_persons(conn) -> list:
    """Return sorted list of persons with roles."""
    rows = conn.execute(
        "SELECT canonical_name, roles FROM persons ORDER BY canonical_name"
    ).fetchall()
    return [dict(r) for r in rows]


def browse_by_person(person_name: str, conn, limit: int = 20) -> list:
    """Return tricks credited to or books by this person."""
    rows = conn.execute(
        """SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.canonical_name) as authors
           FROM tricks t
           JOIN books b ON t.book_id = b.id
           LEFT JOIN book_persons bp ON b.id = bp.book_id
           LEFT JOIN persons p ON bp.person_id = p.id
           WHERE LOWER(t.credited_to) LIKE ? OR LOWER(p.canonical_name) LIKE ?
           GROUP BY t.id
           LIMIT ?""",
        (f"%{person_name.lower()}%", f"%{person_name.lower()}%", limit),
    ).fetchall()
    return [dict(r) for r in rows]
