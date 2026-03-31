#!/usr/bin/env python3
"""
PDF-to-JSON converter for UFPI "Atos da Reitoria" documents.

Handles multiple PDF text layout formats via line-by-line scanning:
  A) Ato number alone on line, date split across next 2 lines
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
    found = []
    for match in re.finditer(r"SIAPE\s+(?:n\.?[º°]?\s*)?(\d{5,8})", text, re.IGNORECASE):
        num = match.group(1)
        if num not in found:
            found.append(num)
    return found


def extract_processes(text: str) -> list[str]:
    found = []
    for match in re.finditer(r"\d{4,5}\.\d{5,6}/\d{4}-\d{2}", text):
        num = match.group(0)
        if num not in found:
            found.append(num)
    return found


def parse_date_to_iso(raw_date: str) -> str:
    raw_date = raw_date.strip()
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", raw_date)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    return raw_date


# ── Phase 1: Detect ato boundaries (READ-ONLY scan, no list mutation) ────────

def detect_ato_boundaries(lines: list[str]) -> list[dict]:
    """
    Scan ALL lines to find ato boundaries. Never skips lines or mutates list.
    Returns list of {"line_idx", "ato_number", "date", "header_end_line", "extra_text"}.
    """
    boundaries = []
    n = len(lines)

    for i in range(n):
        stripped = lines[i].strip()
        if not stripped:
            continue

        # ── Format A: standalone number, then date on next lines ──
        if re.fullmatch(r"\d{1,4}", stripped):
            ato_num = int(stripped)

            if i + 1 >= n:
                continue
            next1 = lines[i + 1].strip()

            # A1: partial date on next line, rest on line after
            #     e.g. "1/2/20" + "25" or "10/7/2" + "025"
            m_partial = re.match(r"^(\d{1,2}/\d{1,2}/\d{1,3})$", next1)
            if m_partial and i + 2 < n:
                rest_line = lines[i + 2].strip()
                m_rest = re.match(r"^(\d{1,3})\b", rest_line)
                if m_rest:
                    date_str = m_partial.group(1) + m_rest.group(1)
                    # Validate reconstructed date has 4-digit year
                    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_str):
                        extra = rest_line[m_rest.end():].strip()
                        boundaries.append({
                            "line_idx": i,
                            "ato_number": ato_num,
                            "date": date_str,
                            "header_end_line": i + 3,
                            "extra_text": extra,
                        })
                        continue

            # A2: full date (+ possible name text) on next line
            m_full = re.match(r"^(\d{1,2}/\d{1,2}/\d{4})\b(.*)", next1)
            if m_full:
                date_str = m_full.group(1)
                extra = m_full.group(2).strip()
                boundaries.append({
                    "line_idx": i,
                    "ato_number": ato_num,
                    "date": date_str,
                    "header_end_line": i + 2,
                    "extra_text": extra,
                })
                continue

            # Not a valid ato boundary, just a random number
            continue

        # ── Format B: number + partial date on same line ──
        #     e.g. "793 5/2/20" + "25" or "2037 11/3/202" + "5"
        m_b = re.match(r"^(\d{1,4})\s+(\d{1,2}/\d{1,2}/\d{1,3})$", stripped)
        if m_b:
            ato_num = int(m_b.group(1))
            date_part = m_b.group(2)

            if i + 1 < n:
                rest_line = lines[i + 1].strip()
                m_rest = re.match(r"^(\d{1,3})\b", rest_line)
                if m_rest:
                    date_str = date_part + m_rest.group(1)
                    # Validate reconstructed date has 4-digit year
                    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_str):
                        extra = rest_line[m_rest.end():].strip()
                        boundaries.append({
                            "line_idx": i,
                            "ato_number": ato_num,
                            "date": date_str,
                            "header_end_line": i + 2,
                            "extra_text": extra,
                        })
                        continue
            continue

        # ── Format C: number + full date (+ text) on same line ──
        m_c = re.match(r"^(\d{1,4})\s+(\d{1,2}/\d{1,2}/\d{4})\b(.*)", stripped)
        if m_c:
            ato_num = int(m_c.group(1))
            date_str = m_c.group(2)
            extra = m_c.group(3).strip()
            boundaries.append({
                "line_idx": i,
                "ato_number": ato_num,
                "date": date_str,
                "header_end_line": i + 1,
                "extra_text": extra,
            })
            continue

    return boundaries


# ── Phase 2: Parse each ato's body ───────────────────────────────────────────

ACTION_VERBS = [
    "Designar", "Conceder", "Autorizar", "Remover", "Dispensar",
    "Nomear", "Constituir", "Substituir", "Exonerar", "Revogar",
    "Retificar", "Prorrogar", "Tornar", "Atualizar", "Cessar",
    "Reverter", "Reconduzir", "Suspender", "Efetivar",
]


def extract_person_from_lines(body_lines: list[str]) -> str:
    name_parts = []
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[Pp]rocesso", stripped):
            break
        if any(stripped.startswith(verb) for verb in ACTION_VERBS):
            break
        if re.match(r"^(\d+\.\s|[-–]\s)", stripped):
            break
        name_parts.append(stripped)

    if name_parts:
        return re.sub(r"\s+", " ", " ".join(name_parts)).strip()
    return ""


def build_description(body_lines: list[str]) -> str:
    started = False
    desc_parts = []

    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not started:
            if re.match(r"^[Pp]rocesso", stripped):
                started = True
            elif any(stripped.startswith(verb) for verb in ACTION_VERBS):
                started = True
            elif re.match(r"^(\d+\.\s|[-–]\s)", stripped):
                started = True
        if started:
            desc_parts.append(stripped)

    return re.sub(r"\s+", " ", " ".join(desc_parts)).strip()


def parse_atos(content_text: str) -> list[dict]:
    lines = content_text.split("\n")
    boundaries = detect_ato_boundaries(lines)

    if not boundaries:
        return []

    atos = []
    for idx, bnd in enumerate(boundaries):
        # Body = lines from header_end_line to the start of the next boundary
        body_start = bnd["header_end_line"]
        if idx + 1 < len(boundaries):
            body_end = boundaries[idx + 1]["line_idx"]
        else:
            body_end = len(lines)

        body_lines = lines[body_start:body_end]

        # Prepend any extra text that was on the date line (e.g. person name)
        if bnd["extra_text"]:
            body_lines = [bnd["extra_text"]] + body_lines

        body_text = "\n".join(body_lines)

        atos.append({
            "ato_number": bnd["ato_number"],
            "date": parse_date_to_iso(bnd["date"]),
            "person": extract_person_from_lines(body_lines),
            "siape": extract_siape(body_text),
            "processes": extract_processes(body_text),
            "description": build_description(body_lines),
        })

    return atos


# ── PDF extraction ───────────────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[str]:
    doc = fitz.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


def convert_pdf_to_json(pdf_path: Path) -> dict:
    year = year_from_path(pdf_path)
    month = month_from_filename(pdf_path.name)
    month_name = MONTH_NUM_TO_NAME.get(month, "") if month else ""

    pages = extract_pages(str(pdf_path))
    description = re.sub(r"\n{3,}", "\n\n", pages[0].strip()) if pages else ""
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
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()

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
