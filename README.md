<h2> Como executar </h3>

1.  Clone o Repositório
2.  Crie um Ambiente Virtual

  É altamente recomendado usar um ambiente virtual para isolar as dependências do projeto e evitar conflitos
  


```

# Criar o ambiente virtual
python -m venv venv

# Ativar o ambiente
# No Windows:
venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate


```

3. Instale as Dependências

```

pip install streamlit python-dotenv langchain langchain-community langchain-openai sentence-transformers faiss-cpu


```


4. Configure sua Chave de API

   Crie um arquivo chamado .env na raiz do projeto.
  
   Adicione sua chave de API ao arquivo da seguinte forma:



```

OPENROUTER_API_KEY="sk-or-v1-sua-chave-secreta-aqui"

```


5. Inicie o servidor do Streamlit com o seguinte comando:

```

streamlit run app.py

```








