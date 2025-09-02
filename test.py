# pdf_extract.py
# Requisitos: pip install pymupdf pytesseract pillow
# Em sistemas sem Tesseract instalado, instale o executável e garanta que está no PATH.
# No Windows, pode ser necessário configurar pytesseract.pytesseract.tesseract_cmd.

import sys
import csv
import io
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def extract_pdf_to_text(pdf_path: Path, out_dir: Path, lang: str = "por+eng") -> Path:
    """
    Extrai texto de um PDF:
      1) tenta get_text() do PyMuPDF por página;
      2) se vier vazio, faz OCR renderizando a página para imagem e usando pytesseract;
    Gera um .txt com texto concatenado e retorna o caminho do TXT.
    """
    txt_out = out_dir / (pdf_path.stem + ".txt")
    pages_csv = out_dir / (pdf_path.stem + "_pages.csv")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_text_parts = []
    rows = []

    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            method = "pymupdf"
            # 1) tentar texto nativo
            text = page.get_text("text")
            if not text or not text.strip():
                # 2) fallback: OCR
                method = "ocr"
                # Renderiza a página em 300 dpi para melhorar OCR
                pix = page.get_pixmap(dpi=300, alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                # Tesseract: idioma português + inglês ajuda em termos técnicos
                text = pytesseract.image_to_string(img, lang=lang)

            all_text_parts.append(f"=== [PDF: {pdf_path.name}] Página {i} ===\n{text}\n")
            rows.append({
                "pdf": pdf_path.name,
                "page": i,
                "method": method,
                "chars": len(text or ""),
            })

    # Escreve TXT consolidado
    txt_out.write_text("\n".join(all_text_parts), encoding="utf-8")

    # Escreve CSV por página
    with pages_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pdf", "page", "method", "chars"])
        w.writeheader()
        w.writerows(rows)

    return txt_out

def main(paths: list[str]):
    if not paths:
        print("Uso: python pdf_extract.py arquivo1.pdf arquivo2.pdf ...")
        sys.exit(1)

    out_dir = Path("saida_texto")
    out_dir.mkdir(exist_ok=True)

    for p in paths:
        pdf_path = Path(p)
        if not pdf_path.exists():
            print(f"[AVISO] Arquivo não encontrado: {pdf_path}")
            continue
        try:
            txt_path = extract_pdf_to_text(pdf_path, out_dir, lang="por+eng")
            print(f"[OK] Texto salvo em: {txt_path}")
        except Exception as e:
            print(f"[ERRO] {pdf_path.name}: {e}")

if __name__ == "__main__":
    main(sys.argv[1:])
