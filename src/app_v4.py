"""
app_v4.py — MagiMatch v4 Gradio UI.

Three tabs: Search, Browse by Effect, Browse by Person.
Dark editorial aesthetic: Playfair Display + IBM Plex Sans, gold accents.

Run:
    PYTHONPATH=. python src/app_v4.py
"""

import gradio as gr
from pathlib import Path

from src.config_loader import paths
from src.db import get_connection, init_schema
from src.logger import get_logger
from src.query_engine import (
    browse_by_effect,
    browse_by_person,
    get_all_effects,
    get_all_persons,
    search,
)

logger = get_logger("app_v4")

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --gold: #c9a84c;
    --gold-dim: #9a7a32;
    --dark-bg: #0d0d0d;
    --card-bg: #161616;
    --border: #2a2a2a;
    --text: #e8e4dc;
    --text-dim: #8a8680;
    --exact-badge: #c9a84c;
    --title-badge: #4a7fb5;
    --semantic-badge: #7b5ea7;
}

body, .gradio-container {
    background: var(--dark-bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: var(--gold) !important;
}

.magimatch-header {
    text-align: center;
    padding: 2rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}

.magimatch-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--gold);
    letter-spacing: 0.05em;
    margin: 0;
}

.magimatch-subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.95rem;
    color: var(--text-dim);
    margin-top: 0.4rem;
    font-weight: 300;
}

.commentary-box {
    border-left: 3px solid var(--gold);
    padding: 0.8rem 1.2rem;
    background: #1a1600;
    margin-bottom: 1.5rem;
    font-style: italic;
    color: var(--text);
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    line-height: 1.6;
    border-radius: 0 4px 4px 0;
}

.result-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.result-card:hover {
    border-color: var(--gold-dim);
}

.trick-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 0.3rem;
}

.trick-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}

.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 2px 7px;
    border-radius: 3px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-right: 6px;
    vertical-align: middle;
}

.badge-exact { background: var(--exact-badge); color: #000; }
.badge-title { background: var(--title-badge); color: #fff; }
.badge-semantic { background: var(--semantic-badge); color: #fff; }

.effect-tag {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid var(--border);
    color: var(--text-dim);
    font-size: 0.72rem;
    padding: 2px 8px;
    border-radius: 3px;
    font-family: 'IBM Plex Sans', sans-serif;
    margin-right: 4px;
}

.trick-description {
    font-size: 0.9rem;
    color: var(--text);
    line-height: 1.55;
    margin-top: 0.5rem;
}

.archive-link {
    display: inline-block;
    margin-top: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: var(--gold);
    text-decoration: none;
}

.archive-link:hover { text-decoration: underline; }

.no-results {
    color: var(--text-dim);
    font-style: italic;
    text-align: center;
    padding: 2rem;
}
"""

# ── Result card HTML ───────────────────────────────────────────────────────────

MATCH_BADGE = {
    "title_exact": '<span class="badge badge-exact">EXACT TITLE</span>',
    "title_fuzzy": '<span class="badge badge-title">TITLE MATCH</span>',
    "semantic":    '<span class="badge badge-semantic">SEMANTIC</span>',
}


def render_card(r: dict) -> str:
    archive_id = r.get("archive_id")
    archive_url = f"https://www.conjuringarchive.com/list/book/{archive_id}" if archive_id else None

    badge = MATCH_BADGE.get(r.get("match_type", "semantic"), MATCH_BADGE["semantic"])
    effect = r.get("effect_category") or ""
    effect_tag = f'<span class="effect-tag">{effect}</span>' if effect else ""
    credited = r.get("credited_to") or ""
    authors = r.get("authors") or ""
    book_title = r.get("book_title") or ""
    year = r.get("pub_year") or ""

    meta_parts = []
    if credited:
        meta_parts.append(f"By {credited}")
    if book_title:
        meta_parts.append(book_title)
    if year:
        meta_parts.append(year)
    if authors:
        meta_parts.append(f"(ed. {authors})")
    meta = " · ".join(meta_parts)

    description = r.get("ai_description") or r.get("description") or ""

    archive_link = ""
    if archive_url:
        archive_link = f'<a class="archive-link" href="{archive_url}" target="_blank">View on Conjuring Archive ↗</a>'

    return f"""
<div class="result-card">
  <p class="trick-title">{r['title']}</p>
  <p class="trick-meta">{meta}</p>
  <div>{badge}{effect_tag}</div>
  <p class="trick-description">{description}</p>
  {archive_link}
</div>
"""


def render_results(results: list, commentary: str) -> str:
    if not results:
        return '<p class="no-results">No results found. Try a different query.</p>'

    html = ""
    if commentary:
        html += f'<div class="commentary-box">{commentary}</div>'

    for r in results:
        html += render_card(r)

    return html


# ── Gradio handlers ────────────────────────────────────────────────────────────

def do_search(query: str, conn):
    if not query or not query.strip():
        return '<p class="no-results">Enter a search query above.</p>'
    try:
        result = search(query.strip(), conn)
        return render_results(result["results"], result["commentary"])
    except Exception as e:
        logger.exception(f"Search error: {e}")
        return f'<p class="no-results">Search error: {e}</p>'


def do_browse_effect(effect: str, conn):
    if not effect:
        return '<p class="no-results">Select an effect category.</p>'
    try:
        results = browse_by_effect(effect, conn, limit=20)
        return render_results(results, f"Showing tricks in the category: {effect}")
    except Exception as e:
        return f'<p class="no-results">Error: {e}</p>'


def do_browse_person(person: str, conn):
    if not person:
        return '<p class="no-results">Select a person.</p>'
    try:
        results = browse_by_person(person, conn, limit=20)
        return render_results(results, f"Showing tricks by or about: {person}")
    except Exception as e:
        return f'<p class="no-results">Error: {e}</p>'


# ── App builder ────────────────────────────────────────────────────────────────

def build_app():
    init_schema(paths.db)
    conn = get_connection(paths.db)

    effects = get_all_effects(conn)
    persons = [p["canonical_name"] for p in get_all_persons(conn)]

    header_html = """
<div class="magimatch-header">
  <h1 class="magimatch-title">MagiMatch</h1>
  <p class="magimatch-subtitle">Semantic search across 126,000+ magic tricks from 3,630 books</p>
</div>
"""

    with gr.Blocks(css=CSS, title="MagiMatch") as demo:
        gr.HTML(header_html)

        with gr.Tabs():
            # ── Search Tab ──────────────────────────────────────────────────
            with gr.Tab("Search"):
                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder='Try: "coin work no gimmicks" or "similar to Matt Mello but underground"',
                        label="Search",
                        scale=5,
                    )
                    search_btn = gr.Button("Search", variant="primary", scale=1)

                results_html = gr.HTML()

                def _search(q):
                    return do_search(q, conn)

                search_btn.click(_search, inputs=query_box, outputs=results_html)
                query_box.submit(_search, inputs=query_box, outputs=results_html)

            # ── Browse by Effect Tab ────────────────────────────────────────
            with gr.Tab("Browse by Effect"):
                effect_dd = gr.Dropdown(choices=effects, label="Effect Category")
                effect_results = gr.HTML()

                def _browse_effect(e):
                    return do_browse_effect(e, conn)

                effect_dd.change(_browse_effect, inputs=effect_dd, outputs=effect_results)

            # ── Browse by Person Tab ────────────────────────────────────────
            with gr.Tab("Browse by Person"):
                person_dd = gr.Dropdown(choices=persons, label="Person", filterable=True)
                person_results = gr.HTML()

                def _browse_person(p):
                    return do_browse_person(p, conn)

                person_dd.change(_browse_person, inputs=person_dd, outputs=person_results)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=True)
