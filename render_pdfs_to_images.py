#!/usr/bin/env python3
"""Render every page of each PDF under a folder to PNG images and log rows in SQLite."""

from __future__ import annotations

import argparse
import hashlib
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF_DIR = BASE_DIR / "Dataset 12"
DEFAULT_IMAGE_ROOT = BASE_DIR / "Dataset 12 images"
DEFAULT_DB = BASE_DIR / "dataset12_page_images.db"


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS page_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_filename TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            pdf_path_rel TEXT,
            pdf_sha256 TEXT,
            pdf_page_count INTEGER NOT NULL,
            page_number INTEGER NOT NULL,
            image_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            image_path_rel TEXT,
            width_px INTEGER NOT NULL,
            height_px INTEGER NOT NULL,
            dpi INTEGER NOT NULL,
            format TEXT NOT NULL DEFAULT 'png',
            image_size_bytes INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            vision_category TEXT,
            vision_description TEXT,
            vision_model TEXT,
            vision_labeled_at TEXT,
            vision_error TEXT,
            UNIQUE(pdf_path, page_number)
        );
        CREATE INDEX IF NOT EXISTS idx_page_images_pdf_filename
            ON page_images(pdf_filename);
        CREATE INDEX IF NOT EXISTS idx_page_images_pdf_path
            ON page_images(pdf_path);
        """
    )
    conn.commit()


def process_pdf(
    pdf_path: Path,
    image_root: Path,
    base_dir: Path,
    dpi: int,
    fmt: str,
    conn: sqlite3.Connection,
    *,
    force: bool,
) -> tuple[int, int]:
    """Return (pages_rendered, pages_skipped)."""
    pdf_path = pdf_path.resolve()
    stem = pdf_path.stem
    out_dir = image_root / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_abs = str(pdf_path)
    pdf_rel = _rel(pdf_path, base_dir)
    filename = pdf_path.name

    doc = fitz.open(pdf_path)
    try:
        if doc.is_encrypted and not doc.authenticate(""):
            print(f"SKIP (encrypted): {filename}", file=sys.stderr)
            return 0, 0
        n = doc.page_count
        if n == 0:
            print(f"SKIP (0 pages): {filename}", file=sys.stderr)
            return 0, 0

        pdf_hash = _sha256_file(pdf_path)
        rendered = 0
        skipped = 0
        now = datetime.now(timezone.utc).isoformat()

        for i in range(n):
            page_number = i + 1
            image_filename = f"page_{page_number:04d}.{fmt}"
            image_path = (out_dir / image_filename).resolve()
            image_abs = str(image_path)
            image_rel = _rel(image_path, base_dir)

            if not force:
                row = conn.execute(
                    "SELECT image_path, image_size_bytes FROM page_images "
                    "WHERE pdf_path = ? AND page_number = ?",
                    (pdf_abs, page_number),
                ).fetchone()
                if row and image_path.is_file() and image_path.stat().st_size == row[1]:
                    skipped += 1
                    continue

            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            if fmt.lower() != "png":
                raise SystemExit(f"Unsupported --format {fmt!r} (only png for now)")
            pix.save(image_abs)

            size_b = image_path.stat().st_size
            w, h = pix.width, pix.height

            conn.execute(
                """
                INSERT INTO page_images (
                    pdf_filename, pdf_path, pdf_path_rel, pdf_sha256, pdf_page_count,
                    page_number, image_filename, image_path, image_path_rel,
                    width_px, height_px, dpi, format, image_size_bytes, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(pdf_path, page_number) DO UPDATE SET
                    pdf_filename = excluded.pdf_filename,
                    pdf_path_rel = excluded.pdf_path_rel,
                    pdf_sha256 = excluded.pdf_sha256,
                    pdf_page_count = excluded.pdf_page_count,
                    image_filename = excluded.image_filename,
                    image_path = excluded.image_path,
                    image_path_rel = excluded.image_path_rel,
                    width_px = excluded.width_px,
                    height_px = excluded.height_px,
                    dpi = excluded.dpi,
                    format = excluded.format,
                    image_size_bytes = excluded.image_size_bytes,
                    created_at = excluded.created_at
                """,
                (
                    filename,
                    pdf_abs,
                    pdf_rel,
                    pdf_hash,
                    n,
                    page_number,
                    image_filename,
                    image_abs,
                    image_rel,
                    w,
                    h,
                    dpi,
                    fmt,
                    size_b,
                    now,
                ),
            )
            rendered += 1
        conn.commit()
        return rendered, skipped
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rasterize PDF pages to PNG and record paths in SQLite."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DEFAULT_PDF_DIR,
        help=f"Directory containing .pdf files (default: {DEFAULT_PDF_DIR.name})",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Root folder for rendered images (default: {DEFAULT_IMAGE_ROOT.name})",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help=f"SQLite database path (default: {DEFAULT_DB.name})",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Render resolution (default 150)")
    parser.add_argument(
        "--format",
        default="png",
        choices=("png",),
        help="Output image format (png only for now)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root for stored relative paths (default: script directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even when DB and image already match",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        print(f"PDF directory not found: {args.pdf_dir}", file=sys.stderr)
        return 1

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {args.pdf_dir}", file=sys.stderr)
        return 1

    args.image_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    try:
        conn.row_factory = sqlite3.Row
        init_db(conn)

        total_r = 0
        total_s = 0
        for j, pdf in enumerate(pdfs, start=1):
            r, s = process_pdf(
                pdf,
                args.image_dir,
                args.base_dir,
                args.dpi,
                args.format,
                conn,
                force=args.force,
            )
            total_r += r
            total_s += s
            print(f"[{j}/{len(pdfs)}] {pdf.name}: rendered {r}, skipped {s}")

        print(f"Done. rendered={total_r} skipped={total_s} db={args.db}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
