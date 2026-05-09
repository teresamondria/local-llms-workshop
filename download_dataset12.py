#!/usr/bin/env python3
"""Download Data Set 12 PDFs from justice.gov using a local filename list.

No HTML scraping: URLs are built as
  https://www.justice.gov/epstein/files/DataSet%2012/<filename>.pdf

Uses the age-verification cookie pattern common on the Epstein subsite.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_LIST = BASE_DIR / "EFTAs.txt"
DEFAULT_OUT = BASE_DIR / "Dataset 12"

# Same path shape as the DOJ site (space encoded as %20).
FILE_BASE = f"https://www.justice.gov/epstein/files/{quote('DataSet 12')}/"

SESSION_COOKIES = {
    "justiceGovAgeVerified": "true",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.justice.gov/epstein/doj-disclosures/data-set-12-files",
}


def load_filenames(list_path: Path) -> list[str]:
    text = list_path.read_text(encoding="utf-8", errors="replace")
    names: list[str] = []
    for line in text.splitlines():
        name = line.strip()
        if not name or not name.lower().endswith(".pdf"):
            continue
        names.append(name)
    return names


def _session() -> requests.Session:
    s = requests.Session()
    for k, v in SESSION_COOKIES.items():
        s.cookies.set(k, v, domain=".justice.gov", path="/")
    return s


def download_one(
    session: requests.Session,
    filename: str,
    out_dir: Path,
    timeout: int,
    retries: int,
) -> tuple[str, bool, str | None]:
    url = f"{FILE_BASE}{filename}"
    dest = out_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return filename, True, "skipped (exists)"

    err: str | None = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=timeout, stream=True)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if "html" in ctype.lower():
                raise RuntimeError(f"got HTML not PDF ({ctype})")

            tmp = dest.with_suffix(".pdf.tmp")
            with open(tmp, "wb") as f:
                first = True
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    if first:
                        if not chunk.startswith(b"%PDF"):
                            raise RuntimeError(f"not a PDF (starts with {chunk[:32]!r})")
                        first = False
                    f.write(chunk)
            tmp.replace(dest)
            return filename, True, None
        except Exception as e:
            err = str(e)
            if dest.with_suffix(".pdf.tmp").exists():
                dest.with_suffix(".pdf.tmp").unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(1.5 * attempt)
    return filename, False, err


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DOJ Epstein Data Set 12 PDFs.")
    parser.add_argument(
        "--list",
        type=Path,
        default=DEFAULT_LIST,
        help=f"Text file with one .pdf filename per line (default: {DEFAULT_LIST.name})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT.name})",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout (s)")
    parser.add_argument("--retries", type=int, default=3, help="Attempts per file")
    args = parser.parse_args()

    if not args.list.exists():
        print(f"List file not found: {args.list}", file=sys.stderr)
        return 1

    names = load_filenames(args.list)
    if not names:
        print(f"No .pdf filenames found in {args.list}", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Files listed: {len(names)}")
    print(f"Output: {args.out}")
    print(f"Base URL: {FILE_BASE}")

    ok = 0
    fail = 0
    # One session per thread (requests.Session is not guaranteed thread-safe).
    def task(fn: str) -> tuple[str, bool, str | None]:
        return download_one(_session(), fn, args.out, args.timeout, args.retries)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(task, fn): fn for fn in names}
        for i, fut in enumerate(as_completed(futures), start=1):
            fn, success, msg = fut.result()
            if success:
                ok += 1
                suffix = f" [{msg}]" if msg else ""
                print(f"[{i}/{len(names)}] OK {fn}{suffix}")
            else:
                fail += 1
                print(f"[{i}/{len(names)}] FAIL {fn}: {msg}", file=sys.stderr)

    print(f"Done. ok={ok} fail={fail}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
