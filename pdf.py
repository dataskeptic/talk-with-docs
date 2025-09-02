# concat_pdfs.py
from pathlib import Path
import sys

try:
    import fitz  # PyMuPDF
except ImportError:
    print("[ERRO] PyMuPDF não instalado. Rode: pip install pymupdf")
    sys.exit(1)

def concat_pdfs_same_folder(output_name: str = "texto_base.txt",
                            log_pages: bool = False) -> tuple[str, int]:
    """
    Lê todos os PDFs na MESMA pasta deste script e escreve um único TXT (UTF-8).
    Retorna (caminho_do_txt, qtd_pdfs). Emite prints de debug no console.
    """
    repo_root = Path(__file__).resolve().parent
    base_dir = repo_root / "documents"
    out_path = base_dir / output_name
    pdf_paths = sorted(base_dir.glob("*.pdf"))

    print(f"[DEBUG] Pasta do script: {base_dir}")
    print(f"[DEBUG] Saída TXT: {out_path}")
    print(f"[DEBUG] PDFs encontrados: {len(pdf_paths)}")
    for p in pdf_paths:
        print(f"  - {p.name}")

    with out_path.open("w", encoding="utf-8") as out:
        for idx, pdf_path in enumerate(pdf_paths, start=1):
            print(f"[DEBUG] ({idx}/{len(pdf_paths)}) Lendo: {pdf_path.name}")
            try:
                with fitz.open(pdf_path) as doc:
                    out.write(f"\n\n=== FILE: {pdf_path.name} ===\n")
                    for i, page in enumerate(doc, start=1):
                        if log_pages:
                            print(f"    - Página {i}")
                        text = page.get_text()  # extração de texto
                        out.write(f"\n--- PAGE {i} ---\n")
                        out.write(text if text else "")
                out.write("\n\n" + "=" * 80 + "\n")
                print(f"[DEBUG] OK: {pdf_path.name}")
            except Exception as e:
                warn = f"[WARN] Falha ao ler {pdf_path.name}: {e}"
                print(warn)
                out.write("\n" + warn + "\n")
                out.write("\n" + "=" * 80 + "\n")

    print(f"[DEBUG] Concluído: {len(pdf_paths)} PDFs processados.")
    print(f"[DEBUG] TXT gerado em: {out_path}")
    return str(out_path), len(pdf_paths)

if __name__ == "__main__":
    # Permite passar um nome de saída opcional: python concat_pdfs.py meu_arquivo.txt
    output_name = sys.argv[13] if len(sys.argv) > 1 else "concatenado.txt"
    concat_pdfs_same_folder(output_name=output_name, log_pages=False)
