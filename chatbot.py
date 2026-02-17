import streamlit as st
import os

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Groq LLM
from langchain_groq import ChatGroq

# ----- Configuration -----
DATA_PATH = "./data"# final fix
CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "mixtral-8x7b-32768"          # Active, fast model (use this or any from list)
# -------------------------

st.set_page_config(page_title="College Chatbot", page_icon="🎓")
st.title("🎓 College Information Assistant")
st.caption("Ask me anything about courses, fees, admissions...")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Load / Create Vector Store ---
@st.cache_resource
def load_vector_store():
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        )
    else:
        loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
    return vectorstore

# --- Build LCEL Chain with Groq ---
@st.cache_resource
def build_chain(_vectorstore):
    # Use the Groq API key from Streamlit secrets
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name=GROQ_MODEL,
        temperature=0.3
    )
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful college assistant. Answer the question based only on the context provided.
    If the answer is not in the context, say you don't know.

    Context: {context}

    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

vectorstore = load_vector_store()
chain = build_chain(vectorstore)

# --- Chat input ---
if prompt := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
