#!/usr/bin/env python3
"""
PDF-to-JSON converter for UFPI "Atos da Reitoria" documents.

Handles multiple PDF text layout formats:
  A) Ato number alone on line, date split across 2 lines
  B) Ato number + partial date on same line, rest on next line
  C) Ato number + full date (+ person start) all on one line

Usage:
    python pdf_to_json.py                       # processes ./documents/ → ./json_output/
    python pdf_to_json.py <input_dir> <out_dir> # custom paths
"""

import fitz  # PyMuPDF
import json
import re
import sys
from pathlib import Path


# ── Month helpers ────────────────────────────────────────────────────────────

MONTH_NAMES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3,
    "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

MONTH_NUM_TO_NAME = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco",
    4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro",
    10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


def month_from_filename(filename: str) -> int | None:
    m = re.match(r"(\d{1,2})-", filename)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 12:
            return num
    return None


def year_from_path(filepath: Path) -> int | None:
    parent = filepath.parent.name
    if re.fullmatch(r"20\d{2}", parent):
        return int(parent)
    return None


# ── Field extractors ─────────────────────────────────────────────────────────

def extract_siape(text: str) -> list[str]:
    """Extract SIAPE registration numbers from text."""
    found = []
    for match in re.finditer(r"SIAPE\s+(?:n\.?[º°]?\s*)?(\d{5,8})", text, re.IGNORECASE):
        num = match.group(1)
        if num not in found:
            found.append(num)
    return found


def extract_processes(text: str) -> list[str]:
    """Extract process numbers like '23111.034471/2024-49'."""
    found = []
    for match in re.finditer(r"\d{4,5}\.\d{5,6}/\d{4}-\d{2}", text):
        num = match.group(0)
        if num not in found:
            found.append(num)
    return found


def parse_date_to_iso(raw_date: str) -> str:
    """Convert M/D/YYYY to ISO YYYY-MM-DD."""
    raw_date = raw_date.strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw_date)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return raw_date


# ── Text normalization ───────────────────────────────────────────────────────

def normalize_text(raw_text: str) -> str:
    """
    Normalize the raw PDF text so that all ato boundaries follow a uniform
    pattern: '<number> <full_date>' on one logical unit.

    Step 1: Reconstruct split dates.
        '1/2/20\\n25'   → '1/2/2025'
        '11/3/202\\n5'  → '11/3/2025'
        '12/1/202\\n5'  → '12/1/2025'

    Step 2: After this normalization, all ato boundaries will match:
        '<number> <M/D/YYYY>'
    regardless of whether they were originally split across lines or not.
    """
    # Join split dates: partial_date\nremaining_digits
    # Matches patterns like "5/2/20\n25" or "11/3/202\n5" or "12/1/202\n5"
    normalized = re.sub(
        r"(\d{1,2}/\d{1,2}/\d{2,3})\s*\n\s*(\d{1,2})\b",
        r"\1\2",
        raw_text,
    )
    return normalized


def find_ato_boundaries(text: str) -> list[tuple[int, int, str, int]]:
    """
    Find all ato boundary positions in the normalized text.

    Returns list of (start_pos, ato_number, date_str, end_of_header_pos).

    Matches patterns like:
        '001 1/2/2025'         (on its own line or mid-line)
        '1223 7/1/2025 JOAO'   (number + date + name on same line)
        '793 5/2/2025'         (after normalization from split date)
    """
    # Pattern: 1-4 digit number, whitespace, then M/D/YYYY date
    pattern = r"(?:^|\n)\s*(\d{1,4})\s+(\d{1,2}/\d{1,2}/\d{4})\b"
    boundaries = []

    for m in re.finditer(pattern, text):
        ato_number = int(m.group(1))
        date_str = m.group(2)
        start_pos = m.start()
        end_pos = m.end()
        boundaries.append((start_pos, ato_number, date_str, end_pos))

    return boundaries


# ── Ato parsing ──────────────────────────────────────────────────────────────

ACTION_VERBS = [
    "Designar", "Conceder", "Autorizar", "Remover", "Dispensar",
    "Nomear", "Constituir", "Substituir", "Exonerar", "Revogar",
    "Retificar", "Prorrogar", "Tornar", "Atualizar", "Cessar",
    "Reverter", "Reconduzir", "Suspender", "Efetivar",
]


def extract_person_from_body(body: str) -> str:
    """
    Extract person name from the beginning of the body text.
    The name is the text before 'Processo' or an action verb.
    """
    lines = body.strip().split("\n")
    name_parts = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Stop at "Processo" line
        if re.match(r"^[Pp]rocesso", stripped):
            break

        # Stop at action verb lines
        if any(stripped.startswith(verb) for verb in ACTION_VERBS):
            break

        # Stop at numbered items like "1. ..." or "- o Processo"
        if re.match(r"^(\d+\.\s|[-–])", stripped):
            break

        name_parts.append(stripped)

    if name_parts:
        name = " ".join(name_parts)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    return ""


def build_description(body: str) -> str:
    """
    Build description from body text, starting from 'Processo' or action verb.
    """
    lines = body.strip().split("\n")
    started = False
    desc_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if started:
                desc_lines.append("")
            continue

        if not started:
            if re.match(r"^[Pp]rocesso", stripped):
                started = True
            elif any(stripped.startswith(verb) for verb in ACTION_VERBS):
                started = True
            elif re.match(r"^(\d+\.\s|[-–])", stripped):
                started = True

        if started:
            desc_lines.append(stripped)

    result = " ".join(desc_lines)
    result = re.sub(r"\s+", " ", result).strip()
    return result


def parse_atos(text: str) -> list[dict]:
    """
    Parse all atos from the full (normalized) text of content pages.
    """
    # Step 1: Normalize split dates
    text = normalize_text(text)

    # Step 2: Find all ato boundaries
    boundaries = find_ato_boundaries(text)

    if not boundaries:
        return []

    # Step 3: Extract each ato's body text
    atos = []
    for i, (start_pos, ato_number, date_str, header_end) in enumerate(boundaries):
        # Body extends from after the header to the start of the next ato
        if i + 1 < len(boundaries):
            body_end = boundaries[i + 1][0]
        else:
            body_end = len(text)

        # The "body" is everything after "number date" up to the next ato
        body = text[header_end:body_end].strip()

        # Extract fields
        person = extract_person_from_body(body)
        siape = extract_siape(body)
        processes = extract_processes(body)
        description = build_description(body)
        date_iso = parse_date_to_iso(date_str)

        atos.append({
            "ato_number": ato_number,
            "date": date_iso,
            "person": person,
            "siape": siape,
            "processes": processes,
            "description": description,
        })

    return atos


# ── PDF extraction ───────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[str]:
    """Extract text from each page of the PDF."""
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def convert_pdf_to_json(pdf_path: Path) -> dict:
    """Convert a single PDF file to our JSON structure."""
    year = year_from_path(pdf_path)
    month = month_from_filename(pdf_path.name)
    month_name = MONTH_NUM_TO_NAME.get(month, "") if month else ""

    pages = extract_pages(str(pdf_path))

    # First page = description (header)
    description = pages[0].strip() if pages else ""
    description = re.sub(r"\n{3,}", "\n\n", description)

    # Remaining pages = ato content
    content = "\n".join(pages[1:]) if len(pages) > 1 else ""
    atos = parse_atos(content)

    return {
        "source_file": str(pdf_path),
        "year": year,
        "month": month,
        "month_name": month_name,
        "description": description,
        "total_atos": len(atos),
        "atos": atos,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.is_dir():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)

    pdf_files = sorted(input_path.rglob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in: {input_dir}")
        sys.exit(1)

    print(f"📂 Found {len(pdf_files)} PDF(s) in {input_dir}")
    print(f"📁 Output directory: {output_dir}\n")

    total_atos = 0
    for pdf_file in pdf_files:
        relative = pdf_file.relative_to(input_path)
        json_name = relative.with_suffix(".json")
        json_dest = output_path / json_name

        print(f"  📄 Processing: {relative} ... ", end="", flush=True)

        try:
            result = convert_pdf_to_json(pdf_file)
            json_dest.parent.mkdir(parents=True, exist_ok=True)

            with open(json_dest, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            n = result["total_atos"]
            total_atos += n
            print(f"✅ {n} atos extracted → {json_name}")

        except Exception as e:
            print(f"❌ Error: {e}")

    print(f"\n{'='*50}")
    print(f"  ✅ Done! {total_atos} total atos from {len(pdf_files)} PDFs")
    print(f"  📁 JSON files saved to: {output_path.resolve()}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        in_dir, out_dir = sys.argv[1], sys.argv[2]
    else:
        in_dir, out_dir = "./documents/", "./json_output/"

    process_directory(in_dir, out_dir)
