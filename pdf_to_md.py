import fitz  # PyMuPDF
import re
import os

def extrair_atos_do_pdf(caminho_pdf):
    """
    Extrai o texto de um PDF e divide-o em atos individuais.
    Retorna uma lista de textos, onde cada texto é um ato.
    """
    try:
        doc = fitz.open(caminho_pdf)
    except Exception as e:
        print(f"Erro ao abrir o PDF {caminho_pdf}: {e}")
        return []

    texto_completo = ""
    for pagina in doc:
        texto_completo += pagina.get_text("text")
    doc.close()

    # Regex para encontrar o início de cada ato.
    # Padrão: um número (ato), seguido de uma data (dd/mm/aaaa).
    # O 'positive lookahead' ((?=\n\d+ \d{1,2}\/\d{1,2}\/\d{4})) garante que a divisão
    # ocorra ANTES do próximo padrão, sem consumi-lo.
    padrao_ato = r'(?=\n\d+\s+\d{1,2}\/\d{1,2}\/\d{4})'
    
    # Tratamento preliminar para casos em que o número do ato está em uma linha
    # e a data na linha seguinte, comum nos seus PDFs.
    texto_completo = re.sub(r'(\n\d+)\n(\d{1,2}\/\d{1,2}\/\d{4})', r'\1 \2', texto_completo)

    atos_brutos = re.split(padrao_ato, texto_completo)
    
    # Filtra entradas vazias que podem surgir da divisão
    return [ato.strip() for ato in atos_brutos if ato.strip()]

def formatar_ato_para_markdown(texto_ato):
    """
    Formata o texto de um único ato para o formato Markdown sugerido.
    """
    # Extrai número do ato e data da primeira linha
    match_cabecalho = re.match(r'(\d+)\s+(\d{1,2}\/\d{1,2}\/\d{4})', texto_ato)
    if not match_cabecalho:
        return None # Ignora trechos que não são atos (ex: cabeçalho do documento)

    num_ato = match_cabecalho.group(1)
    data_ato = match_cabecalho.group(2)
    
    # Extrai número(s) do processo
    processos = re.findall(r'Processo n.º\s+([\d./-]+)', texto_ato)
    processos_str = ", ".join(processos) if processos else "Não identificado"
    
    # Limpa o corpo do texto do ato
    # Remove a linha do cabeçalho e as linhas de processo já extraídas
    corpo_ato = re.sub(r'^\d+\s+\d{1,2}\/\d{1,2}\/\d{4}\s*', '', texto_ato, count=1)
    corpo_ato = re.sub(r'Processo n.º\s+[\d./-]+\s*', '', corpo_ato)
    corpo_ato = corpo_ato.strip()

    # Monta o Markdown final
    markdown_formatado = (
        f"# Ato {num_ato}\n\n"
        f"**Data do Ato:** {data_ato}\n"
        f"**Processo(s):** {processos_str}\n\n"
        f"{corpo_ato}\n\n"
        "---\n"
    )
    
    return markdown_formatado

def processar_pdfs_em_pasta(pasta_entrada, arquivo_saida):
    """
    Processa todos os arquivos PDF em uma pasta e salva o resultado em um único arquivo Markdown.
    """
    with open(arquivo_saida, 'w', encoding='utf-8') as f_out:
        # Lista todos os arquivos na pasta de entrada
        arquivos_pdf = [f for f in os.listdir(pasta_entrada) if f.lower().endswith('.pdf')]

        for nome_arquivo in arquivos_pdf:
            caminho_completo = os.path.join(pasta_entrada, nome_arquivo)
            print(f"Processando arquivo: {nome_arquivo}...")
            
            atos = extrair_atos_do_pdf(caminho_completo)
            
            if not atos:
                print(f"Nenhum ato encontrado em {nome_arquivo}.")
                continue

            for ato_bruto in atos:
                ato_formatado = formatar_ato_para_markdown(ato_bruto)
                if ato_formatado:
                    f_out.write(ato_formatado)
            
            print(f"Arquivo {nome_arquivo} processado com sucesso. {len(atos)} atos encontrados.")

# --- Como Usar ---
if __name__ == "__main__":
    # Coloque seus PDFs em uma pasta, por exemplo, 'meus_pdfs'
    pasta_dos_pdfs = "./documents/"  # Usando o diretório atual
    nome_arquivo_markdown = "atos_compilados.md"
    
    # Garante que a pasta existe
    if not os.path.isdir(pasta_dos_pdfs):
        print(f"Erro: A pasta '{pasta_dos_pdfs}' não foi encontrada.")
    else:
        processar_pdfs_em_pasta(pasta_dos_pdfs, nome_arquivo_markdown)
        print(f"\nConversão concluída! Os atos foram salvos em '{nome_arquivo_markdown}'.")

