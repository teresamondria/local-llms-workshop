#!/usr/bin/env python3
"""Label page images as 'photo' or 'document' using a multimodal Ollama model.

Vision models accept base64 PNG/JPEG via POST /api/chat (see Ollama docs).
Text-only models (e.g. llama3.1:8b) cannot see images.

Default model is llava (fast baseline). Other options:

  ollama pull llava
      -> use --model llava

  ollama pull llama3.2-vision
      -> use --model llama3.2-vision  (11B-class; avoid :90b on laptops)

  ollama pull qwen2.5vl:7b
      # or: ollama pull qwen2.5vl:3b   (lighter; still strong on documents)
      -> use --model qwen2.5vl:7b
  In the Ollama library the current Qwen vision line is "qwen2.5vl" (Qwen2.5-VL);
  tags vary—check https://ollama.com/library/qwen2.5vl

Requires a recent Ollama (qwen2.5vl notes Ollama 0.7.0+ in its readme).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "dataset12_page_images.db"
DEFAULT_CSV = BASE_DIR / "vision_labels.csv"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llava"

# Columns exported to CSV (vision + identifiers for each page image).
CSV_EXPORT_SQL = """
    SELECT id, pdf_filename, page_number, pdf_path, pdf_path_rel,
           image_filename, image_path, image_path_rel,
           width_px, height_px, dpi,
           vision_category, vision_description, vision_model, vision_labeled_at, vision_error
    FROM page_images
    ORDER BY pdf_filename, page_number
"""

USER_PROMPT = """You are examining a single page image from a public document release. Look carefully at the whole page.

First classify the page into exactly one category:
- "photo": the page is primarily a photograph, snapshot, portrait, scene, or other pictorial image (not dominated by continuous lines of text).
- "document": the page is primarily typed or printed text, a form, letter, legal filing, table, memo, cover sheet, or scan where lines of text or form fields dominate (even if a small photo or logo appears).

Then write a thorough, concrete description of everything you can see. Be specific and observational, not speculative. Cover as many of the following as apply:
- Layout: margins, columns, headers, footers, letterhead, stamps, watermarks, holes or folds.
- Text: approximate density, typewriter vs printed vs handwriting, bullet lists, tables, visible headings or labels (quote short phrases if readable).
- Non-text graphics: photographs, signatures, seals, diagrams, charts, redactions, highlights, handwriting blocks.
- Visual quality: scan contrast, skew, cropping, glare, monochrome vs color.
- If faces or identifiable people appear, describe only what is visible (e.g. "group portrait of several adults in suits") without naming real individuals unless printed on the page.

Write multiple sentences (aim for at least 5–10 sentences when the page is complex). The description must stay inside one JSON string; use spaces between sentences (no raw line breaks inside the string).

Reply with ONLY valid JSON, no markdown, no code fences, in exactly this shape:
{"category":"photo","description":"Your detailed description here."}
or
{"category":"document","description":"Your detailed description here."}

Rules: "category" must be exactly lowercase photo or document. The "description" must be non-empty. Do not use double-quote characters inside the description text (use single quotes or rephrase) so the JSON stays valid."""


def export_vision_csv(db_path: Path, csv_path: Path) -> int:
    """Write all page_images rows (with vision columns) to UTF-8 CSV. Returns row count."""
    conn = sqlite3.connect(db_path)
    try:
        ensure_vision_columns(conn)
        cur = conn.execute(CSV_EXPORT_SQL)
        colnames = [d[0] for d in cur.description]
        rows = cur.fetchall()
    finally:
        conn.close()

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(colnames)
        w.writerows(rows)
    return len(rows)


def ensure_vision_columns(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(page_images)")
    existing = {row[1] for row in cur.fetchall()}
    additions = [
        ("vision_category", "TEXT"),
        ("vision_description", "TEXT"),
        ("vision_model", "TEXT"),
        ("vision_labeled_at", "TEXT"),
        ("vision_error", "TEXT"),
    ]
    for col, decl in additions:
        if col not in existing:
            conn.execute(f"ALTER TABLE page_images ADD COLUMN {col} {decl}")
    conn.commit()


def image_to_b64_png(path: Path, max_edge: int) -> str:
    im = Image.open(path)
    im = im.convert("RGB")
    w, h = im.size
    if max_edge > 0 and max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def parse_model_json(text: str) -> tuple[str, str]:
    """Return (category, description) or raise ValueError."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    data: Any = json.loads(t)
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    cat = str(data.get("category", "")).strip().lower()
    desc = str(data.get("description", "")).strip()
    if cat not in ("photo", "document"):
        raise ValueError(f"category must be photo or document, got {cat!r}")
    if not desc:
        raise ValueError("empty description")
    return cat, desc


def ollama_chat_vision(
    ollama_url: str,
    model: str,
    b64_png: str,
    timeout: int,
) -> str:
    url = f"{ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": USER_PROMPT,
                "images": [b64_png],
            }
        ],
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    msg = body.get("message") or {}
    content = msg.get("content")
    if not content:
        raise RuntimeError(f"unexpected response: {body!r}")
    return str(content).strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify page PNGs as photo vs document via Ollama vision."
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Vision Ollama model. Default: llava. Alternatives: llama3.2-vision, qwen2.5vl:7b, "
            "qwen2.5vl:3b (see ollama.com/library). Do not use llama3.1:8b."
        ),
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=1280,
        help="Max width/height in pixels before resize (0 = no resize). Saves VRAM/time.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Ollama request timeout in seconds (long descriptions need more time).",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between requests")
    parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-label even if already labeled")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Write/update this CSV after labeling (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV after labeling",
    )
    parser.add_argument(
        "--export-csv-only",
        action="store_true",
        help="Only export CSV from the database (no Ollama calls)",
    )
    args = parser.parse_args()

    if not args.db.is_file():
        print(f"Database not found: {args.db}", file=sys.stderr)
        return 1

    if args.export_csv_only:
        n = export_vision_csv(args.db, args.csv)
        print(f"Wrote {n} rows to {args.csv}")
        return 0

    # Quick health check
    try:
        tags = requests.get(
            f"{args.ollama_url.rstrip('/')}/api/tags", timeout=5
        )
        tags.raise_for_status()
    except requests.RequestException as e:
        print(f"Cannot reach Ollama at {args.ollama_url}: {e}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    try:
        ensure_vision_columns(conn)
        q = """
            SELECT id, image_path, vision_category
            FROM page_images
            ORDER BY pdf_filename, page_number
        """
        rows = conn.execute(q).fetchall()
    finally:
        conn.close()

    processed = 0
    skipped = 0
    failed = 0
    conn = sqlite3.connect(args.db)

    try:
        for row in rows:
            if args.limit and processed >= args.limit:
                break
            pid, image_path, existing_cat = row[0], row[1], row[2]
            path = Path(image_path)
            if not path.is_file():
                failed += 1
                conn.execute(
                    "UPDATE page_images SET vision_error = ? WHERE id = ?",
                    (f"missing file: {image_path}", pid),
                )
                conn.commit()
                print(f"MISSING id={pid} {image_path}", file=sys.stderr)
                continue

            if existing_cat and not args.force:
                skipped += 1
                continue

            try:
                b64 = image_to_b64_png(path, args.max_edge)
                raw = ollama_chat_vision(
                    args.ollama_url, args.model, b64, args.timeout
                )
                cat, desc = parse_model_json(raw)
                now = datetime.now(timezone.utc).isoformat()
                conn.execute(
                    """
                    UPDATE page_images SET
                        vision_category = ?,
                        vision_description = ?,
                        vision_model = ?,
                        vision_labeled_at = ?,
                        vision_error = NULL
                    WHERE id = ?
                    """,
                    (cat, desc, args.model, now, pid),
                )
                conn.commit()
                processed += 1
                preview = desc[:120] + ("…" if len(desc) > 120 else "")
                print(f"[{processed}] id={pid} {path.name} -> {cat}: {preview}")
            except Exception as e:
                failed += 1
                err = str(e)[:500]
                conn.execute(
                    "UPDATE page_images SET vision_error = ? WHERE id = ?",
                    (err, pid),
                )
                conn.commit()
                print(f"FAIL id={pid} {path.name}: {e}", file=sys.stderr)

            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        conn.close()

    print(f"Done. labeled={processed} skipped={skipped} failed={failed}")
    if not args.no_csv:
        n = export_vision_csv(args.db, args.csv)
        print(f"CSV: {n} rows -> {args.csv}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
