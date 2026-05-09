#!/usr/bin/env python3
"""
Generate embeddings for existing vision descriptions in dataset12_page_images.db
and store them back in SQLite for fast local similarity search.

This uses Ollama's embeddings endpoint (POST /api/embeddings) with a text embedding model
such as `nomic-embed-text`.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "dataset12_page_images.db"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "nomic-embed-text"


def ensure_embedding_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(page_images)")
    existing = {row[1] for row in cur.fetchall()}
    additions = [
        ("desc_embedding_json", "TEXT"),
        ("desc_embedding_dim", "INTEGER"),
        ("desc_embedding_model", "TEXT"),
        ("desc_embedded_at", "TEXT"),
        ("desc_embedding_error", "TEXT"),
    ]
    for col, decl in additions:
        if col not in existing:
            conn.execute(f"ALTER TABLE page_images ADD COLUMN {col} {decl}")
    conn.commit()


def ollama_embed(ollama_url: str, model: str, text: str, timeout: int) -> list[float]:
    url = f"{ollama_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    body: Any = r.json()
    emb = body.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"unexpected embeddings response: {body!r}")
    out: list[float] = []
    for x in emb:
        try:
            out.append(float(x))
        except Exception as e:
            raise RuntimeError(f"non-numeric embedding value: {x!r}") from e
    if not math.isfinite(out[0]):
        raise RuntimeError("embedding contains non-finite values")
    return out


def chunk_preview(s: str, n: int = 110) -> str:
    s = " ".join((s or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def iter_rows(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    q = """
        SELECT
            id,
            vision_description,
            desc_embedding_json
        FROM page_images
        WHERE vision_description IS NOT NULL
        ORDER BY pdf_filename, page_number
    """
    cur = conn.execute(q)
    for row in cur:
        yield row


def main() -> int:
    p = argparse.ArgumentParser(
        description="Embed vision_description strings via Ollama and store vectors in SQLite."
    )
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Ollama embedding model")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    p.add_argument("--limit", type=int, default=0, help="Max rows to embed (0 = all)")
    p.add_argument("--force", action="store_true", help="Re-embed even if embedding exists")
    args = p.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    # Health check
    try:
        tags = requests.get(f"{args.ollama_url.rstrip('/')}/api/tags", timeout=5)
        tags.raise_for_status()
    except requests.RequestException as e:
        print(f"Cannot reach Ollama at {args.ollama_url}: {e}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    try:
        ensure_embedding_columns(conn)
        rows = list(iter_rows(conn))
    finally:
        conn.close()

    processed = 0
    skipped = 0
    failed = 0

    conn = sqlite3.connect(args.db)
    try:
        conn.row_factory = sqlite3.Row
        ensure_embedding_columns(conn)

        for row in rows:
            if args.limit and processed >= args.limit:
                break

            rid = int(row["id"])
            desc = str(row["vision_description"] or "").strip()
            has_emb = bool(row["desc_embedding_json"])

            if not desc:
                skipped += 1
                continue
            if has_emb and not args.force:
                skipped += 1
                continue

            try:
                emb = ollama_embed(args.ollama_url, args.model, desc, args.timeout)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    UPDATE page_images SET
                        desc_embedding_json = ?,
                        desc_embedding_dim = ?,
                        desc_embedding_model = ?,
                        desc_embedded_at = ?,
                        desc_embedding_error = NULL
                    WHERE id = ?
                    """,
                    (json.dumps(emb), len(emb), args.model, now, rid),
                )
                conn.commit()
                processed += 1
                print(f"[{processed}] id={rid} dim={len(emb)} {chunk_preview(desc)}")
            except Exception as e:
                failed += 1
                err = str(e)[:500]
                conn.execute(
                    "UPDATE page_images SET desc_embedding_error = ? WHERE id = ?",
                    (err, rid),
                )
                conn.commit()
                print(f"FAIL id={rid}: {e}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        conn.close()

    print(f"Done. embedded={processed} skipped={skipped} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

