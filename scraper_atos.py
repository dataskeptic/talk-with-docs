#!/usr/bin/env python3
"""
UFPI Atos da Reitoria Scraper
Downloads monthly PDF/DOCX files from https://www.ufpi.br/atos-e-spds
for a user-specified number of past years, organized newest to oldest.
"""

import os
import re
import sys
import time
import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, unquote
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://www.ufpi.br/atos-e-spds"
MONTH_ORDER = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3,
    "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

MONTH_PT = {
    1: "01-Janeiro", 2: "02-Fevereiro", 3: "03-Marco",
    4: "04-Abril",   5: "05-Maio",      6: "06-Junho",
    7: "07-Julho",   8: "08-Agosto",    9: "09-Setembro",
    10: "10-Outubro", 11: "11-Novembro", 12: "12-Dezembro",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_page(url: str) -> str:
    """Fetch page HTML with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            print(f"  ⚠  Attempt {attempt+1}/3 failed: {exc}")
            time.sleep(2)
    raise RuntimeError(f"Could not fetch page: {url}")


def detect_year_from_context(text: str) -> int | None:
    """Extract a 4-digit year from a string."""
    m = re.search(r"\b(20\d{2}|19\d{2})\b", text)
    return int(m.group(1)) if m else None


def month_from_text(text: str) -> int | None:
    """Return numeric month from a Portuguese month name inside text."""
    text_lower = text.lower()
    for name, num in MONTH_ORDER.items():
        if name in text_lower:
            return num
    return None


def parse_entries(html: str) -> list[dict]:
    """
    Parse the page and return a list of:
        {"year": int, "month": int, "label": str, "url": str}
    """
    soup = BeautifulSoup(html, "html.parser")
    entries = []
    current_year = None

    # The page content lives inside article / div.item-page or similar
    body = soup.find("div", class_=re.compile(r"item-page|article__body|entry-content", re.I))
    if body is None:
        body = soup.find("article") or soup.find("main") or soup.body

    # Walk every element in document order
    for elem in body.descendants:
        if elem.name is None:
            continue  # skip NavigableString that is not a tag

        text = elem.get_text(strip=True)

        # Year header: a text node / heading / paragraph that is ONLY a year
        if elem.name in ("h2", "h3", "h4", "p", "strong", "b", "span"):
            m = re.fullmatch(r"(20\d{2}|19\d{2})", text)
            if m:
                current_year = int(m.group(1))
                continue

        # Anchor with a file link
        if elem.name == "a" and elem.get("href"):
            href = elem["href"].strip()
            # Only interested in file links (pdf / docx)
            if not re.search(r"\.(pdf|docx?|PDF|DOCX?)(\?.*)?$", href, re.I):
                continue

            url = urljoin("https://www.ufpi.br", href)
            link_text = text

            month_num = month_from_text(link_text)

            # If year is not yet detected, try to infer from surrounding text or URL
            year = current_year
            if year is None:
                year = detect_year_from_context(link_text) or detect_year_from_context(href)

            if month_num and year:
                entries.append({
                    "year": year,
                    "month": month_num,
                    "label": link_text or f"{year}-{month_num:02d}",
                    "url": url,
                })

    # Deduplicate (same year+month keeps the last URL found, which is usually
    # the most-complete one — mirrors the site's own pattern for duplicates)
    seen: dict[tuple, dict] = {}
    for e in entries:
        key = (e["year"], e["month"])
        seen[key] = e  # last one wins

    result = list(seen.values())
    result.sort(key=lambda x: (x["year"], x["month"]), reverse=True)
    return result


def download_file(url: str, dest_path: Path) -> bool:
    """Download a single file to dest_path. Returns True on success."""
    if dest_path.exists():
        print(f"    ✔  Already downloaded: {dest_path.name}")
        return True
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_kb = dest_path.stat().st_size // 1024
        print(f"    ✔  {dest_path.name}  ({size_kb} KB)")
        return True
    except requests.RequestException as exc:
        print(f"    ✗  FAILED {url} → {exc}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  UFPI – Atos da Reitoria  |  Downloader")
    print("=" * 60)

    # --- Ask how many years ---
    current_year = datetime.date.today().year
    while True:
        try:
            n = int(input(f"\nHow many of the last years do you want to download? (1–{current_year - 2011}): ").strip())
            if 1 <= n <= current_year - 2011:
                break
            print(f"  Please enter a number between 1 and {current_year - 2011}.")
        except ValueError:
            print("  Please enter a valid integer.")

    target_years = list(range(current_year - n + 1, current_year + 1))
    print(f"\n  → Will download years: {target_years[0]}–{target_years[-1]}")

    # --- Optional: custom output directory ---
    default_dir = Path("atos_ufpi")
    dir_input = input(f"\nOutput directory [default: {default_dir}]: ").strip()
    output_dir = Path(dir_input) if dir_input else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  → Saving to: {output_dir.resolve()}")

    # --- Fetch and parse ---
    print("\n  Fetching page …")
    html = fetch_page(BASE_URL)
    all_entries = parse_entries(html)
    print(f"  Found {len(all_entries)} total entries on the page.")

    # --- Filter by selected years ---
    selected = [e for e in all_entries if e["year"] in target_years]
    # Sort: newest first (year DESC, month DESC)
    selected.sort(key=lambda x: (x["year"], x["month"]), reverse=True)

    if not selected:
        print("\n  No files found for the selected years. Exiting.")
        sys.exit(0)

    print(f"  {len(selected)} files match the selected years (newest → oldest).\n")

    # --- Download ---
    success, fail = 0, 0
    for i, entry in enumerate(selected, 1):
        year_folder = output_dir / str(entry["year"])
        ext = Path(urlparse(entry["url"]).path).suffix or ".pdf"
        fname = f"{MONTH_PT[entry['month']]}{ext}"
        dest = year_folder / fname

        print(f"  [{i:03d}/{len(selected):03d}] {entry['year']} – {MONTH_PT[entry['month']]}")
        ok = download_file(entry["url"], dest)
        if ok:
            success += 1
        else:
            fail += 1
        time.sleep(0.3)  

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  Done!  {success} downloaded,  {fail} failed.")
    print(f"  Files are in: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
