import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

load_dotenv()

st.set_page_config(
    page_title="RAG com Arctic Embed e DeepSeek",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🚀 Sistema RAG com Snowflake Arctic Embed")

with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input(
        "🔑 Chave da API OpenRouter",
        type="password",
        value=os.getenv("OPENROUTER_API_KEY", "")
    )
    model_id = st.selectbox(
        "🤖 Modelo LLM",
        ["deepseek/deepseek-chat-v3.1:free", "meta-llama/llama-3.2-8b-instruct:free", "google/gemma-2-9b-it:free"],
        index=0
    )
    temperature = st.slider("🌡️ Temperatura", 0.0, 2.0, 0.2)

class ArcticEmbeddings(Embeddings):
    def __init__(self, device="cpu"):
        self.model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
        self.model.to(device)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], prompt_name="query", normalize_embeddings=True)[0].tolist()

#  Carregar o modelo e o índice com cache para performance
@st.cache_resource
def load_embeddings_model():
    return ArcticEmbeddings(device="cpu")

@st.cache_resource
def load_faiss_index(_embeddings):
    index_path = "faiss_index"
    if not os.path.isdir(index_path):
        return None
    return FAISS.load_local(index_path, _embeddings, allow_dangerous_deserialization=True)

embeddings_model = load_embeddings_model()
vector_store = load_faiss_index(embeddings_model)

if vector_store is None:
    st.error("Pasta 'faiss_index_arctic_1024' não encontrada. Certifique-se de descompactar o arquivo do Colab aqui.")
    st.stop()

# Configurar o retriever com MMR na sidebar
with st.sidebar:
    st.subheader("🔍 Configurações de Busca")
    k = st.slider("Resultados Finais (k)", 3, 15, 6, help="Número de documentos que o LLM verá.")
    fetch_k = st.slider("Documentos para MMR (fetch_k)", 20, 100, 50, help="Número de documentos pré-selecionados para diversificar.")
    lambda_mult = st.slider("Diversidade MMR (0=max, 1=min)", 0.0, 1.0, 0.5, help="Controle entre relevância e diversidade.")

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
)

#  Configurar o LLM e a cadeia RAG
llm = ChatOpenAI(
    openai_api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model_name=model_id,
    temperature=temperature,
    max_tokens=1500,
)
rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Interface principal
st.info("A base de conhecimento foi carregada com sucesso. Faça sua pergunta abaixo.")
question = st.text_input("💬 Sua pergunta:", placeholder="Ex: Quais são os principais temas do documento?")

if st.button("Perguntar", type="primary"):
    if not api_key:
        st.warning("Por favor, insira sua chave da API OpenRouter na barra lateral.")
    elif not question:
        st.warning("Por favor, digite uma pergunta.")
    else:
        with st.spinner("🧠 Consultando a base de conhecimento..."):
            try:
                result = rag_chain.invoke({"query": question})
                st.markdown("### Resposta")
                st.success(result['result'])
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")

st.markdown("---")
st.caption("Sistema RAG com Embeddings `Snowflake Arctic` | Chunking por Tokens | Retriever MMR")
