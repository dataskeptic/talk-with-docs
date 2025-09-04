import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="RAG UFPI - Atos da Reitoria",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 RAG para Atos da Reitoria da UFPI")
st.markdown("Consulte os atos oficiais da UFPI usando os modelos de embedding da Snowflake e LLMs do OpenRouter.")

# --- INTERFACE DA SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configurações")
    model_id = st.selectbox(
        "🤖 Modelo LLM",
    ["deepseek/deepseek-chat-v3.1:free", "meta-llama/llama-3.2-8b-instruct:free", "google/gemma-2-9b-it:free", "google/gemini-2.5-flash-image-preview:free", "openai/gpt-oss-120b:free", "openai/gpt-oss-20b:free", "qwen/qwen3-coder:free", "moonshotai/kimi-k2:free", "google/gemma-3n-e2b-it:free", "google/gemma-3-flash-e2b-it:free"],
        index=0,
        help="Escolha o modelo de linguagem que irá gerar as respostas."
    )
    temperature = st.slider("🌡️ Temperatura", 0.0, 1.5, 0.2, help="Define a criatividade do modelo.")

    st.header("🔍 Busca e Filtragem")
    processo_filter = st.text_input(
        "🔢 Filtrar por Nº do Processo",
        placeholder="Ex: 23111.015169/2022-28",
        help="Digite um ou mais números de processo, separados por vírgula."
    )
    k = st.slider("Resultados Finais (k)", 3, 15, 6, help="Número de documentos que o LLM usará.")
    fetch_k = st.slider("Documentos Pré-selecionados (fetch_k)", 20, 100, 50, help="Documentos recuperados para MMR.")
    lambda_mult = st.slider("Diversidade (lambda)", 0.0, 1.0, 0.5, help="0.0 para relevância, 1.0 para diversidade.")

# --- LÓGICA DE CARREGAMENTO COM CACHE ---
class ArcticEmbeddings(Embeddings):
    """Wrapper customizado para o modelo Snowflake Arctic."""
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", device=device)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], prompt_name="query", normalize_embeddings=True)[0].tolist()

# **CORREÇÃO 1: Melhorando a lógica de cache para a mensagem não persistir**
# Inicializa o estado da sessão para controlar a mensagem de carregamento
if 'embeddings_loaded' not in st.session_state:
    st.session_state.embeddings_loaded = False

@st.cache_resource
def load_components():
    """Função única para carregar todos os componentes pesados."""
    
    embeddings = ArcticEmbeddings(device="cpu")
    
    index_path = "faiss_index"  # Nome correto da pasta
    if not os.path.isdir(index_path):
        st.error(f"Pasta do índice ('{index_path}') não encontrada. Certifique-se de descompactar o arquivo do Colab aqui.")
        return None, None
        
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
    # Atualiza o estado da sessão após o carregamento
    st.session_state.embeddings_loaded = True
    return embeddings, vector_store

embeddings_model, vector_store = load_components()

if vector_store is None:
    st.stop()

# --- LÓGICA DE BUSCA E MONTAGEM DA CADEIA ---
filter_list = [p.strip() for p in processo_filter.split(',')] if processo_filter else []
search_filter = {"processos": filter_list} if filter_list else {}

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult, "filter": search_filter}
)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.warning("A chave da API OpenRouter não foi encontrada no arquivo `.env`.")
    st.stop()

llm = ChatOpenAI(
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model_name=model_id,
    temperature=temperature,
    max_tokens=2048,
)

# **CORREÇÃO 2: Alinhando a chave de input para 'input'**
# O `create_retrieval_chain` espera uma chave 'input' por padrão.
prompt_template = """
Você é um assistente especializado em analisar documentos oficiais da Universidade Federal do Piauí (UFPI).
Responda à pergunta do usuário com base EXCLUSIVAMENTE no contexto fornecido abaixo.
Se a informação não estiver no contexto, diga "A informação não foi encontrada nos atos fornecidos.".
Seja claro, objetivo e cite os números dos atos e processos que basearam sua resposta.

**Contexto:**
{context}

**Pergunta:**
{input}

**Resposta Detalhada:**
"""
prompt = ChatPromptTemplate.from_template(prompt_template)
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# --- INTERFACE PRINCIPAL ---
question = st.text_input(
    "💬 Faça sua pergunta sobre os atos da reitoria:",
    placeholder="Ex: Quem foi nomeado para o cargo de Assistente em Administração em junho de 2022?"
)

if st.button("Analisar Atos", type="primary"):
    if not question:
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("🧠 Consultando a base de conhecimento e gerando a resposta..."):
            try:
                # **CORREÇÃO 2 (continuação): Passando a pergunta com a chave 'input'**
                response = retrieval_chain.invoke({"input": question})
                
                st.markdown("### Resposta")
                st.info(response['answer'])

                if 'context' in response and response['context']:
                    st.markdown("---")
                    st.markdown("### 📄 Fontes Utilizadas")
                    for doc in response['context']:
                        metadata = doc.metadata
                        ato = metadata.get('Ato', 'N/A')
                        data_ato = metadata.get('data_ato', 'N/A')
                        processos = ', '.join(metadata.get('processos', [])) or 'Não identificado'
                        
                        with st.expander(f"**Ato {ato}** - Data: {data_ato}"):
                            st.caption(f"Processos: {processos}")
                            st.write(doc.page_content)
                            
            except Exception as e:
                st.error(f"Ocorreu um erro ao processar sua pergunta: {e}")
                st.error(f"Detalhes do erro: {str(e)}")

st.markdown("---")
st.caption("Desenvolvido com LangChain, Streamlit e modelos de IA de ponta.")
