#!/usr/bin/env python3
"""
Very small demo UI:
- turns a query into an embedding (Ollama)
- finds the most similar *image descriptions* (cosine similarity)
- shows the corresponding page image + the stored description
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import gradio as gr
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BASE_DIR / "dataset12_page_images.db"
DEFAULT_OLLAMA = "http://127.0.0.1:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@dataclass(frozen=True)
class Item:
    id: int
    pdf_filename: str
    page_number: int
    image_path: str
    description: str
    embedding: list[float]


def ollama_embed(ollama_url: str, model: str, text: str, timeout: int = 60) -> list[float]:
    url = f"{ollama_url.rstrip('/')}/api/embeddings"
    payload = {"model": model, "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    body: Any = r.json()
    emb = body.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise RuntimeError(f"unexpected embeddings response: {body!r}")
    return [float(x) for x in emb]


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def load_items(db_path: Path) -> list[Item]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        q = """
            SELECT
                id, pdf_filename, page_number, image_path,
                vision_description, desc_embedding_json
            FROM page_images
            WHERE
                vision_description IS NOT NULL
                AND desc_embedding_json IS NOT NULL
            ORDER BY pdf_filename, page_number
        """
        rows = conn.execute(q).fetchall()
    finally:
        conn.close()

    items: list[Item] = []
    for r in rows:
        try:
            emb = json.loads(r["desc_embedding_json"])
            if not isinstance(emb, list) or not emb:
                continue
            items.append(
                Item(
                    id=int(r["id"]),
                    pdf_filename=str(r["pdf_filename"] or ""),
                    page_number=int(r["page_number"] or 0),
                    image_path=str(r["image_path"] or ""),
                    description=str(r["vision_description"] or ""),
                    embedding=[float(x) for x in emb],
                )
            )
        except Exception:
            continue
    return items


def main() -> int:
    p = argparse.ArgumentParser(description="Tiny UI: search image descriptions by embeddings.")
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    if not args.db.is_file():
        raise SystemExit(f"DB not found: {args.db}")

    # Preload items once (fast + simple for demo).
    items = load_items(args.db)
    if not items:
        raise SystemExit(
            "No embedded descriptions found. Run embed_image_descriptions_ollama.py first."
        )

    def search(query: str) -> tuple[list[tuple[str, str]], str]:
        q = (query or "").strip()
        if not q:
            return [], "Type a query to search."

        q_emb = ollama_embed(args.ollama_url, args.embed_model, q)
        scored = [(cosine(q_emb, it.embedding), it) for it in items]
        scored.sort(key=lambda x: x[0], reverse=True)

        top = scored[: max(1, args.top_k)]
        gallery: list[tuple[str, str]] = []
        descriptions: list[str] = []
        for score, it in top:
            caption = f"{it.pdf_filename} p.{it.page_number} — score {score:.3f}"
            gallery.append((it.image_path, caption))
            descriptions.append(f"{caption}\n\n{it.description}".strip())
        selected = descriptions[0] if descriptions else ""
        return gallery, json.dumps(descriptions), selected

    with gr.Blocks(title="Dataset 12 demo search") as demo:
        gr.Markdown("## Search page images by description embeddings")
        q = gr.Textbox(label="Query", placeholder="e.g., 'handwritten note' or 'court filing'")
        btn = gr.Button("Search")
        gallery = gr.Gallery(label="Top matches", columns=2, rows=3, height=380)
        _descs_json = gr.State("[]")
        selected = gr.Textbox(label="Selected description", lines=10, interactive=False)

        def on_select(descs_json: str, evt: gr.SelectData) -> str:
            try:
                descs = json.loads(descs_json or "[]")
                if not isinstance(descs, list):
                    return ""
                i = int(getattr(evt, "index", 0) or 0)
                if 0 <= i < len(descs):
                    return str(descs[i])
                return ""
            except Exception:
                return ""

        btn.click(search, inputs=[q], outputs=[gallery, _descs_json, selected])
        q.submit(search, inputs=[q], outputs=[gallery, _descs_json, selected])
        gallery.select(on_select, inputs=[_descs_json], outputs=[selected])

    demo.launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

