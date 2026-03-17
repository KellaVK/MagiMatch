"""
embedder.py — Step 5: Embed all tricks and save as .npy arrays.

Usage:
    PYTHONPATH=. python src/embedder.py
    PYTHONPATH=. python src/embedder.py --resume
    PYTHONPATH=. python src/embedder.py --reset
"""

import argparse
import numpy as np
from pathlib import Path

from src.checkpoint import Checkpoint
from src.config_loader import cfg, paths
from src.db import get_connection
from src.logger import get_logger
from src.openai_client import embed_texts

logger = get_logger("embedder")

BATCH_SIZE = cfg.embeddings.batch_size
MODEL = cfg.embeddings.model


def run(resume: bool = False, reset: bool = False):
    emb_dir = paths.embeddings
    emb_dir.mkdir(parents=True, exist_ok=True)

    tricks_npy = emb_dir / "tricks.npy"
    ids_npy = emb_dir / "trick_ids.npy"
    checkpoint = Checkpoint("embedder", paths.checkpoints)

    if reset:
        logger.info("Reset: removing existing embeddings and checkpoint.")
        for f in [tricks_npy, ids_npy]:
            if f.exists():
                f.unlink()
        checkpoint.reset()

    conn = get_connection(paths.db)

    # Load all tricks that have embed_text
    rows = conn.execute(
        "SELECT id, embed_text FROM tricks WHERE embed_text IS NOT NULL ORDER BY id"
    ).fetchall()
    conn.close()

    logger.info(f"Found {len(rows)} tricks with embed_text.")

    trick_ids = [r["id"] for r in rows]
    texts = [r["embed_text"] for r in rows]

    # Check resume state
    start_batch = 0
    existing_embs = []
    if resume and checkpoint.get_meta("last_batch") is not None:
        start_batch = checkpoint.get_meta("last_batch") + 1
        if tricks_npy.exists() and ids_npy.exists():
            existing_embs = list(np.load(str(tricks_npy)))
            logger.info(f"Resuming from batch {start_batch} ({len(existing_embs)} embeddings already done)")

    all_embeddings = existing_embs
    processed_ids = trick_ids[: len(existing_embs)]

    remaining_ids = trick_ids[len(existing_embs):]
    remaining_texts = texts[len(existing_embs):]

    total_batches = (len(remaining_texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(remaining_texts), BATCH_SIZE):
        batch_texts = remaining_texts[i : i + BATCH_SIZE]
        batch_ids = remaining_ids[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE

        logger.info(f"Embedding batch {batch_num+1}/{total_batches} ({len(batch_texts)} texts)...")
        embs = embed_texts(batch_texts, model=MODEL, batch_size=BATCH_SIZE)
        all_embeddings.extend(embs)
        processed_ids.extend(batch_ids)

        # Save incrementally
        np.save(str(tricks_npy), np.array(all_embeddings, dtype=np.float32))
        np.save(str(ids_npy), np.array(processed_ids, dtype=np.int64))
        checkpoint.set_meta("last_batch", start_batch + batch_num)

    logger.info(f"Embeddings complete. Shape: {np.array(all_embeddings).shape}")
    logger.info(f"Saved to: {tricks_npy}, {ids_npy}")

    # Log to DB
    conn = get_connection(paths.db)
    conn.execute(
        "INSERT INTO embedding_versions (entity_type, model_name, record_count) VALUES (?, ?, ?)",
        ("trick", MODEL, len(all_embeddings)),
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MagiMatch Step 5 — Embedding generator")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    run(resume=args.resume, reset=args.reset)
