"""
query_engine.py — MagiMatch v5 Query Engine (patched).

Pipeline:
  1. Intent parsing (gpt-4o-mini)
  2. Plot-alias expansion
  3. Person style expansion (profiles dict → Serper → DB fallback)
  4. Title pre-pass (SQL exact + fuzzy on book AND trick titles)
  5. Semantic search (cosine similarity over tricks.npy)
  6. Merge + deduplicate (with required_keywords + propless hard-filter)
  7. AI descriptions (batched gpt-4o-mini)
  8. Commentary (gpt-4o-mini)

Fixes vs v5 original:
  - require_no_props: props column doesn't exist → now expands query +
    injects prop terms into excluded_keywords
  - Title query early return now applies full merge_results filtering
  - required_keywords: new intent field; hard-filters results missing the prop
  - MAGIC_PLOT_ALIASES: expands plot-description queries (e.g. "trick that
    cannot be explained") into technique keywords
  - title_search: now also searches trick titles, not just book titles
  - get_style_description: fixed DB column names (canonical_name, no style_summary)
  - Regex safety net for "similar to X" / "like X" patterns in parse_intent
  - generate_descriptions: more robust JSON parsing
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.db import get_connection
from src.logger import get_logger
from src.openai_client import chat_completion, embed_single

logger = get_logger("query_engine")


# ── Magician profiles ──────────────────────────────────────────────────────────
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

# Runtime style cache
_style_cache: dict = {}


# ── Magic plot aliases ─────────────────────────────────────────────────────────
# Maps descriptive phrases / informal plot names → rich search terms.
# Matched case-insensitively against the raw query.
MAGIC_PLOT_ALIASES: dict[str, str] = {
    # Classic "impossible" plots
    "trick that cannot be explained": (
        "out of this world paul curry spectator sorts cards red black "
        "inexplicable self-working hands-off impossible classic plot"
    ),
    "the trick that cannot be explained": (
        "out of this world paul curry spectator sorts cards red black "
        "inexplicable self-working hands-off impossible classic"
    ),
    "out of this world": (
        "out of this world paul curry spectator sorts red black cards "
        "impossible hands-off self-working classic plot"
    ),
    # Classic plots by description
    "ambitious card": (
        "ambitious card rising to top classic repeat vanish appear card control"
    ),
    "four aces": (
        "four ace assembly trick production gathering revelation ace location aces"
    ),
    "coins across": (
        "coins across travel han ping chien multiple coins transposition "
        "coin vanish appear migration"
    ),
    "card in wallet": "card to wallet transposition transportation surprise revelation",
    "bill in lemon": "signed bill in lemon fruit transposition signed currency penetration",
    "color change": "visual color change transformation top change packet color",
    "anniversary waltz": "anniversary waltz cards matching mates romantic pairs revelation",
    "slow motion four aces": "slow motion four aces rene lavand one hand assembly patience",
    "chicago opener": "red backed card Chicago opener revelation surprise ending red card",
    "two card monte": "two card monte three card transposition deceptive visual",
    "living dead": "living dead test mentalism billet work telepathy",
    "pick a card": "card selection force peek glimpse classic card trick any card",
    "rising card": "card rises from deck mechanical elevator card magic",
    "card to impossible location": "card to impossible location transportation penetration surprise",
    "collectors": "collectors four cards sandwich collectors plot",
    "triumph": "triumph face up face down shuffle restoration spectator chaos",
    "twisting the aces": "twisting aces Ed Ullis packet face up face down flip rotation",
    "oil and water": "oil and water red black cards separate interlace mix",
    "reset": "reset Paul Harris aces four card transposition instant reset",
    "slow motion aces": "slow motion aces Vernon classic assembly deceptive natural",
    "the matrix": "matrix Al Schneider coins under cards travel vanish appear",
    "coins through table": "coins through table penetration solid through solid coin magic",
    "sponge balls": "sponge ball vanish appear multiply comedy classic children",
    "linking rings": "linking rings chinese rings link unlink solid metal penetration",
    "cups and balls": "cups and balls classic three shells penetration vanish appear final load",
}

# ── Propless / no-props support ────────────────────────────────────────────────
# Appended to the semantic query when require_no_props=True.
PROPLESS_EXPANSION = (
    "propless no props empty hands pure psychological mental "
    "no cards no coins no objects no deck thought impression "
    "verbal force suggestion bare hands invisible nothing"
)

# These prop terms are added to excluded_keywords when require_no_props=True,
# hard-filtering tricks whose descriptions mention physical objects.
PROPLESS_EXCLUDED_PROPS = [
    "card", "deck", "coin", "rope", "wand", "ball", "silk",
    "sponge", "envelope", "wallet", "billet", "pen",
    "phone", "watch", "ring", "handkerchief", "paper", "cup",
]


# ── Serper web lookup ──────────────────────────────────────────────────────────

def _serper_lookup(magician_name: str) -> Optional[str]:
    """Use Serper to find a magician's style online. Returns raw snippet text."""
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
        snippets = [item["snippet"] for item in data.get("organic", [])[:5] if item.get("snippet")]
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


def get_style_description(name: str, conn, web_enriched: list) -> str:
    """
    Return a keyword-rich style description for the named magician.
    Appends the name to web_enriched list if Serper was used.
    """
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
            web_enriched.append(name)
            return summary
        except Exception as e:
            logger.warning(f"GPT style summary failed for {name}: {e}")

    # 3. DB fallback: infer from credited tricks
    # NOTE: persons table has canonical_name (not name) and no style_summary column.
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

# Regex safety net for "similar to X" / "like X" patterns.
# Used AFTER GPT parsing to ensure referenced_persons is always populated.
_SIMILAR_PATTERNS = [
    re.compile(r"\bsimilar\s+to\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
    re.compile(r"\blike\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+style\s+of\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
]

_EXCLUDED_PATTERNS = [
    re.compile(r"\bnot\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
    re.compile(r"\bexclude\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
    re.compile(r"\bno\s+([A-Z][a-zA-Z .'-]+)", re.IGNORECASE),
]


def _regex_extract_persons(query: str):
    """
    Regex safety net: extract referenced and excluded persons from the raw query.
    Returns (referenced: list[str], excluded: list[str]).
    Only used to SUPPLEMENT (not override) GPT parsing.
    """
    referenced = []
    for pat in _SIMILAR_PATTERNS:
        for m in pat.finditer(query):
            name = m.group(1).strip().rstrip(",.")
            if len(name.split()) <= 4:  # guard against runaway captures
                referenced.append(name)

    excluded = []
    for pat in _EXCLUDED_PATTERNS:
        for m in pat.finditer(query):
            name = m.group(1).strip().rstrip(",.")
            if len(name.split()) <= 4:
                excluded.append(name)

    return referenced, excluded


def parse_intent(query: str) -> dict:
    """
    Parse user query into structured intent using gpt-4o-mini.

    Returns:
        topic, keywords, excluded_persons, referenced_persons, is_title_query,
        require_no_props, excluded_keywords, required_keywords, title
    """
    system = """You parse magic trick search queries into structured JSON.
Return ONLY valid JSON with these fields:

- topic: string, the core topic in plain language (used for commentary, not search)

- embedding_query: string, a SHORT, keyword-dense phrase (8-15 words max) optimized for
  vector similarity search — NO stop words, NO filler ("a", "the", "where", "that", "and",
  "is", "for", "with"), just the most specific nouns, verbs, and adjectives that describe
  what the user wants. Convert conversational phrasing into precise magic terminology.
  Examples:
    "a card trick where the spectator shuffles and the card is located"
      → "spectator shuffles deck card location found selected card playing cards"
    "something like what Slydini does with coins"
      → "coin magic misdirection natural movement lap work visual"
    "tricks from Expert at the Card Table"
      → "Expert at the Card Table Erdnase gambling moves sleights"
    "propless mentalism not Matt Mello"
      → "propless mentalism psychological bare hands no objects"

- keywords: list of strings, important search keywords taken directly from the query

- excluded_persons: list of strings, names the user wants EXPLICITLY EXCLUDED (e.g. "not Vernon", "not Asi Wind")

- referenced_persons: list of strings, magicians whose STYLE is used as a model (e.g. "like Mello", "similar to Asi Wind") — do NOT add these to excluded_persons; they go here only

- is_title_query: boolean, true if user is searching for a specific book or trick title

- title: string or null, the specific book or trick title being searched — extract only the title text, omitting surrounding words like "tricks from" or "about" — only set when is_title_query is true

- require_no_props: boolean, true ONLY when user explicitly wants effects with absolutely NO physical objects whatsoever — triggered ONLY by: "propless", "no props", "no objects", "empty handed", "bare hands" — do NOT set true for "impromptu", "no gimmicks", or other qualifiers

- excluded_keywords: list of strings, concepts that must NOT appear in results — extract the root word (e.g. "no gimmicks" → ["gimmick"], "no cards" → ["card"], "no coins" → ["coin"], "no ropes" → ["rope"])

- required_keywords: list of strings, specific prop types or concepts that MUST appear in results — extract root word when user is clearly specific about prop type (e.g. "coin tricks" → ["coin"], "rope magic" → ["rope"], "sponge ball" → ["sponge"]) — leave empty for general queries or when prop type is part of style not requirement

IMPORTANT: Be conservative. Only extract what the user explicitly stated. Do not infer or add themes they did not mention.
"""
    prompt = f'Parse this magic search query: "{query}"'
    defaults = {
        "topic": query,
        "embedding_query": query,
        "keywords": [],
        "excluded_persons": [],
        "referenced_persons": [],
        "is_title_query": False,
        "title": None,
        "require_no_props": False,
        "excluded_keywords": [],
        "required_keywords": [],
    }
    try:
        raw = chat_completion(prompt, system=system, max_tokens=600, temperature=0.1, json_mode=True)
        result = json.loads(raw)
        parsed = {
            "topic": result.get("topic", query),
            # embedding_query falls back to topic, then raw query if GPT omits it
            "embedding_query": result.get("embedding_query") or result.get("topic") or query,
            "keywords": result.get("keywords", []),
            "excluded_persons": [p.strip().lower() for p in result.get("excluded_persons", [])],
            "referenced_persons": [p.strip() for p in result.get("referenced_persons", [])],
            "is_title_query": bool(result.get("is_title_query", False)),
            "title": result.get("title") or None,
            "require_no_props": bool(result.get("require_no_props", False)),
            "excluded_keywords": [k.strip().lower() for k in result.get("excluded_keywords", [])],
            "required_keywords": [k.strip().lower() for k in result.get("required_keywords", [])],
        }
    except Exception as e:
        logger.warning(f"Intent parsing failed: {e}")
        parsed = defaults.copy()

    # Regex safety net: supplement GPT result with pattern-matched persons
    regex_ref, regex_excl = _regex_extract_persons(query)
    existing_ref_lower = {p.lower() for p in parsed["referenced_persons"]}
    for name in regex_ref:
        if name.lower() not in existing_ref_lower:
            parsed["referenced_persons"].append(name)
            existing_ref_lower.add(name.lower())

    existing_excl = set(parsed["excluded_persons"])
    for name in regex_excl:
        nl = name.lower()
        # Don't add to excluded if already in referenced
        if nl not in existing_ref_lower and nl not in existing_excl:
            parsed["excluded_persons"].append(nl)
            existing_excl.add(nl)

    return parsed


# ── Plot alias expansion ───────────────────────────────────────────────────────

def apply_plot_aliases(query: str) -> str:
    """
    If the query matches a known plot alias (case-insensitive substring),
    append the alias's expansion to the query so semantic search finds the
    right content even when descriptions don't use the user's exact phrasing.
    """
    q_lower = query.lower()
    expansions = []
    for phrase, expansion in MAGIC_PLOT_ALIASES.items():
        if phrase in q_lower:
            expansions.append(expansion)
    if expansions:
        return query + " " + " ".join(expansions)
    return query


# ── Title pre-pass ─────────────────────────────────────────────────────────────

_TITLE_SELECT = """
    SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
           t.embed_text,
           b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
           GROUP_CONCAT(DISTINCT p.name) as authors
    FROM tricks t
    JOIN books b ON t.book_id = b.id
    LEFT JOIN book_persons bp ON b.id = bp.book_id AND bp.role = 'author'
    LEFT JOIN persons p ON bp.person_id = p.id
"""


def title_search(query: str, conn, title_override: Optional[str] = None) -> list:
    """
    Search for tricks by book title or trick title.

    Steps:
      1. Exact match on book title
      2. Fuzzy (word-by-word LIKE) on book title
      3. Fuzzy on trick title (if no book title results)

    Args:
        query: raw user query (used if title_override is None)
        title_override: extracted title string from intent parser (preferred)
    """
    search_text = (title_override or query).strip()
    results: list = []
    seen_ids: set = set()

    # ── Pass 1: Exact book title match ────────────────────────────────────────
    rows = conn.execute(
        _TITLE_SELECT + """
        WHERE LOWER(b.title) = LOWER(?)
        GROUP BY t.id
        LIMIT 15""",
        (search_text,),
    ).fetchall()
    for r in rows:
        if r["id"] not in seen_ids:
            results.append({**dict(r), "match_type": "title_exact"})
            seen_ids.add(r["id"])
    if results:
        return results

    # ── Pass 2: Fuzzy book title match ────────────────────────────────────────
    words = [w for w in re.split(r"\s+", search_text) if len(w) >= 3]
    if words:
        like_clauses = " AND ".join(f"LOWER(b.title) LIKE ?" for _ in words)
        params = [f"%{w.lower()}%" for w in words]
        rows = conn.execute(
            _TITLE_SELECT + f"""
            WHERE {like_clauses}
            GROUP BY t.id
            LIMIT 15""",
            params,
        ).fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                results.append({**dict(r), "match_type": "title_fuzzy"})
                seen_ids.add(r["id"])
    if results:
        return results

    # ── Pass 3: Fuzzy trick title match (for specific effect lookups) ─────────
    if words:
        like_clauses = " AND ".join(f"LOWER(t.title) LIKE ?" for _ in words)
        params = [f"%{w.lower()}%" for w in words]
        rows = conn.execute(
            _TITLE_SELECT + f"""
            WHERE {like_clauses}
            GROUP BY t.id
            LIMIT 15""",
            params,
        ).fetchall()
        for r in rows:
            if r["id"] not in seen_ids:
                results.append({**dict(r), "match_type": "title_fuzzy"})
                seen_ids.add(r["id"])

    return results


# ── Semantic search ────────────────────────────────────────────────────────────

def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1e-9
    query_norm = np.linalg.norm(query_vec) or 1e-9
    return matrix @ query_vec / (norms * query_norm)


def semantic_search(
    query_text: str,
    embeddings: np.ndarray,
    trick_ids: np.ndarray,
    top_k: int = 40,
    min_score: float = 0.22,
) -> list:
    """
    Embed query, compute cosine similarity, return top_k (trick_id, score) pairs.

    min_score: cosine similarity floor — results below this are silently dropped.
    Prevents low-confidence matches from surfacing and forcing the AI to
    rationalize irrelevant results. 0.22 is a conservative threshold for
    OpenAI text-embedding-3-small; raise to 0.28 if results are still noisy.
    """
    query_vec = embed_single(query_text)
    scores = cosine_similarity_matrix(query_vec, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (int(trick_ids[i]), float(scores[i]))
        for i in top_indices
        if float(scores[i]) >= min_score
    ]


# ── Fetch trick details ────────────────────────────────────────────────────────

def fetch_tricks(trick_ids: list, conn) -> dict:
    """Fetch full trick+book+author rows. Returns dict keyed by trick_id."""
    if not trick_ids:
        return {}
    placeholders = ",".join("?" * len(trick_ids))
    rows = conn.execute(
        f"""SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
                  t.embed_text,
                  b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                  GROUP_CONCAT(DISTINCT p.name) as authors
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
    excluded_keywords: list = None,
    required_keywords: list = None,
    require_no_props: bool = False,
    max_per_book: int = 2,
    max_per_author: int = 3,
    total: int = 15,
) -> list:
    """
    Merge title and semantic results, deduplicate, apply per-book/author caps.
    Filters: excluded_persons, excluded_keywords, required_keywords, require_no_props.

    NOTE: require_no_props hard-filtering is done upstream (PROPLESS_EXCLUDED_PROPS
    are injected into excluded_keywords in search() before calling here).
    The flag is kept for any future props-column support.
    """
    seen_ids: set = set()
    book_counts: dict = {}
    author_counts: dict = {}
    final: list = []

    excluded_lower = [e.lower() for e in (excluded_persons or [])]
    excluded_kw = [k.lower() for k in (excluded_keywords or [])]
    required_kw = [k.lower() for k in (required_keywords or [])]

    def _haystack(r) -> str:
        return " ".join([
            (r.get("title") or ""),
            (r.get("description") or ""),
            (r.get("embed_text") or ""),
            (r.get("effect_category") or ""),
        ]).lower()

    def _is_filtered(r) -> bool:
        """Return True if this trick should be excluded."""
        credited = (r.get("credited_to") or "").lower()
        authors = (r.get("authors") or "").lower()

        # Hard-exclude explicitly named persons (substring match handles partial names)
        if any(ex in credited or ex in authors for ex in excluded_lower):
            return True

        hay = _haystack(r)

        # Excluded keywords: must NOT appear anywhere in the trick text
        if excluded_kw and any(kw in hay for kw in excluded_kw):
            return True

        # Required keywords: ALL must appear somewhere in the trick text
        if required_kw and not all(kw in hay for kw in required_kw):
            return True

        return False

    def _add(result_dict, match_type, score=0.0):
        tid = result_dict["id"]
        if tid in seen_ids:
            return
        if _is_filtered(result_dict):
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

    # Semantic results in score order
    for tid, score in semantic_results:
        if len(final) >= total:
            break
        if tid not in trick_details:
            continue
        _add(trick_details[tid], "semantic", score)

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
            f"   Book: {r.get('book_title', '')}\n"
            f"   Description: {r.get('description', '') or '(none)'}\n"
            f"   Category: {r.get('effect_category', '') or '(unknown)'}\n"
            f"   Credited to: {r.get('credited_to', '') or '(unknown)'}"
        )

    prompt = (
        "For each magic trick below, write 1-2 sentences explaining what it IS and why a magician "
        "might want it. Be specific. If the description is vague, explain what that technique involves. "
        "Return ONLY a JSON array of strings, one per trick, in order.\n\n"
        + "\n\n".join(items)
    )
    max_tok = min(1500, max(200, len(results) * 120))
    try:
        raw = chat_completion(prompt, max_tokens=max_tok, temperature=0.4, json_mode=True)
        parsed = json.loads(raw)
        # Robust extraction: handle both list and dict responses
        if isinstance(parsed, dict):
            # Try common dict shapes: {"descriptions": [...]} or {0: ..., 1: ...}
            for key in ("descriptions", "results", "tricks", "items"):
                if key in parsed and isinstance(parsed[key], list):
                    descs = parsed[key]
                    break
            else:
                descs = list(parsed.values())
                # If values are all strings, use them; otherwise flatten
                if descs and isinstance(descs[0], list):
                    descs = descs[0]
        else:
            descs = parsed  # already a list
        for i, r in enumerate(results):
            r["ai_description"] = str(descs[i]) if i < len(descs) and descs[i] else (r.get("description") or "")
    except Exception as e:
        logger.warning(f"AI descriptions failed: {e}")
        for r in results:
            r["ai_description"] = r.get("description") or ""
    return results


# ── Commentary ─────────────────────────────────────────────────────────────────

def generate_commentary(query: str, results: list) -> str:
    """2-3 sentence commentary explaining why these results match the query."""
    if not results:
        return "No results found for this query."
    titles = [f"{r['title']} ({r.get('book_title', '')})" for r in results[:5]]
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


# ── Result card builder ────────────────────────────────────────────────────────

def build_result_card(result: dict, rank: int) -> dict:
    """
    Convert a raw search result into the dict expected by the Gradio UI render_card().
    """
    archive_id = result.get("archive_id")
    return {
        "trick_title":     result.get("title", "Unknown"),
        "effect_category": result.get("effect_category"),
        "author":          result.get("credited_to") or result.get("authors") or "Unknown",
        "book_title":      result.get("book_title", "Unknown"),
        "pub_year":        result.get("pub_year") or "",
        "ai_description":  result.get("ai_description", ""),
        "raw_description": result.get("description") or "",
        "match_type":      result.get("match_type", "semantic"),
        "archive_url": (
            f"https://www.conjuringarchive.com/list/medium/{archive_id}"
            if archive_id else None
        ),
        "rank":  rank,
        "score": round(result.get("score", 0.0), 3),
    }


# ── QueryEngine class ──────────────────────────────────────────────────────────

class QueryEngine:
    """
    Main MagiMatch search engine.

    Usage:
        engine = QueryEngine(
            db_path="data/processed/magimatch.db",
            embeddings_dir="data/embeddings",
            openai_api_key="sk-...",
        )
        response = engine.search("propless mentalism not Matt Mello")
    """

    def __init__(
        self,
        db_path: str,
        embeddings_dir: str,
        openai_api_key: Optional[str] = None,
    ):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self.db_path = Path(db_path)
        self.embeddings_dir = Path(embeddings_dir)

        self.conn = get_connection(self.db_path)

        emb_path = self.embeddings_dir / "tricks.npy"
        ids_path = self.embeddings_dir / "trick_ids.npy"
        if emb_path.exists() and ids_path.exists():
            self.embeddings = np.load(str(emb_path), allow_pickle=True)
            self.trick_ids = np.load(str(ids_path), allow_pickle=True)
            print(f"✅ Loaded {len(self.trick_ids):,} embeddings ({self.embeddings.shape[1]}d)")
        else:
            self.embeddings = None
            self.trick_ids = None
            print("⚠️  Embeddings not found — semantic search disabled. Run src/embedder.py first.")

    def search(self, query: str, top_k: int = 15, describe_count: Optional[int] = 3) -> dict:
        """
        Full search pipeline.

        Args:
            top_k: total results to fetch and return.
            describe_count: how many results to generate AI descriptions for immediately.
                            Pass None to describe all results.

        Returns:
            {
                "results": list of result dicts (pass each to build_result_card),
                "commentary": str,
                "web_enriched_persons": list of names looked up via Serper,
                "query_info": parsed intent dict,
            }
        """
        web_enriched: list = []

        # 1. Parse intent
        intent = parse_intent(query)
        logger.info(f"Intent: {intent}")

        # 2. Start from the keyword-dense embedding_query (not the raw topic/query).
        #    apply_plot_aliases expands known plot phrases into technique keywords.
        expanded_query = apply_plot_aliases(intent["embedding_query"])

        # 3. Expand style for referenced persons
        for person in intent["referenced_persons"]:
            style = get_style_description(person, self.conn, web_enriched)
            expanded_query = f"{expanded_query} {style}"

        # Auto-exclude the referenced persons themselves from results.
        all_excluded = intent["excluded_persons"] + [
            p.strip().lower() for p in intent["referenced_persons"]
        ]

        # 4. Handle require_no_props
        #    The tricks table has no `props` column, so we use two approaches:
        #    a) Expand the semantic query with propless keywords (soft signal)
        #    b) Inject physical-prop root words into excluded_keywords (hard filter)
        excluded_keywords = list(intent.get("excluded_keywords", []))

        if intent.get("require_no_props"):
            expanded_query = f"{expanded_query} {PROPLESS_EXPANSION}"
            # Hard-filter: exclude results that mention common prop types.
            # Use a set to avoid duplicates with user-specified excluded_keywords.
            existing_excl = set(excluded_keywords)
            for prop in PROPLESS_EXCLUDED_PROPS:
                if prop not in existing_excl:
                    excluded_keywords.append(prop)

        required_keywords = intent.get("required_keywords", [])

        # 5. Title pre-pass
        title_results = []
        title_override = intent.get("title")  # cleaned title extracted by GPT
        if intent["is_title_query"] or len(query.split()) <= 5:
            title_results = title_search(query, self.conn, title_override=title_override)

            if title_results and intent["is_title_query"]:
                # Early return for explicit title queries — but APPLY FULL FILTERING first.
                filtered_title = merge_results(
                    title_results=title_results,
                    semantic_results=[],
                    trick_details={},
                    excluded_persons=all_excluded,
                    excluded_keywords=excluded_keywords,
                    required_keywords=required_keywords,
                    require_no_props=intent.get("require_no_props", False),
                    max_per_book=top_k,       # for title queries let through all from that book
                    max_per_author=top_k,
                    total=top_k,
                )
                n = describe_count if describe_count is not None else len(filtered_title)
                described = generate_descriptions(filtered_title[:n])
                rest = filtered_title[n:]
                all_results = described + rest
                commentary = generate_commentary(query, described)
                return {
                    "results": all_results,
                    "commentary": commentary,
                    "web_enriched_persons": web_enriched,
                    "query_info": intent,
                }

        # 6. Semantic search (fetch extra candidates to survive filtering)
        sem_results = []
        if self.embeddings is not None:
            sem_results = semantic_search(
                expanded_query, self.embeddings, self.trick_ids, top_k=max(top_k * 5, 60)
            )

        # 7. Fetch details for semantic candidates
        sem_ids = [tid for tid, _ in sem_results]
        trick_details = fetch_tricks(sem_ids, self.conn)

        # 8. Merge + deduplicate + filter
        merged = merge_results(
            title_results=title_results,
            semantic_results=sem_results,
            trick_details=trick_details,
            excluded_persons=all_excluded,
            excluded_keywords=excluded_keywords,
            required_keywords=required_keywords,
            require_no_props=intent.get("require_no_props", False),
            total=top_k,
        )

        # 9. AI descriptions (lazy: only describe_count upfront)
        n = describe_count if describe_count is not None else len(merged)
        described = generate_descriptions(merged[:n])
        rest = merged[n:]
        all_results = described + rest

        # 10. Commentary
        commentary = generate_commentary(query, described)

        return {
            "results": all_results,
            "commentary": commentary,
            "web_enriched_persons": web_enriched,
            "query_info": intent,
        }

    def enrich_results(self, results: list) -> list:
        """
        Generate AI descriptions for results that don't have them yet.
        Used by the load-more UI to lazily generate descriptions on demand.
        """
        to_describe = [r for r in results if not r.get("ai_description")]
        if not to_describe:
            return results
        generate_descriptions(to_describe)
        return results

    def get_all_effects(self) -> list:
        """Return sorted list of distinct effect categories."""
        rows = self.conn.execute(
            "SELECT DISTINCT effect_category FROM tricks WHERE effect_category IS NOT NULL ORDER BY effect_category"
        ).fetchall()
        return [r["effect_category"] for r in rows]

    def browse_by_effect(self, effect: str, limit: int = 20) -> list:
        """Return tricks matching an effect category."""
        rows = self.conn.execute(
            """SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
                      b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                      GROUP_CONCAT(DISTINCT p.name) as authors
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

    def browse_by_person(self, person_name: str, limit: int = 20) -> list:
        """Return tricks credited to or books by this person."""
        rows = self.conn.execute(
            """SELECT t.id, t.title, t.description, t.effect_category, t.credited_to,
                      b.id as book_id, b.title as book_title, b.pub_year, b.archive_id,
                      GROUP_CONCAT(DISTINCT p.name) as authors
               FROM tricks t
               JOIN books b ON t.book_id = b.id
               LEFT JOIN book_persons bp ON b.id = bp.book_id
               LEFT JOIN persons p ON bp.person_id = p.id
               WHERE LOWER(t.credited_to) LIKE ? OR LOWER(p.name) LIKE ?
               GROUP BY t.id
               LIMIT ?""",
            (f"%{person_name.lower()}%", f"%{person_name.lower()}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]
