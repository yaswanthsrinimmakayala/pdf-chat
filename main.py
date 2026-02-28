import os
import logging
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,AIMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
st.title("Chat about your PDF")
st.caption("Ask anything about the uploaded file")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF",type="pdf")
    if uploaded_file is None:
        st.info("Please upload a file")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
CHROMA_DB = os.path.join(os.path.dirname(tmp_path),"chroma_db")

@st.cache_resource
def load_vectorstore(pdf_pth):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  
    if os.path.exists(CHROMA_DB):
        logging.info("Loaded existing vectorstore")
        return Chroma(persist_directory=CHROMA_DB,embedding_function=embeddings)
    # opening the PDF
    loader = PyPDFLoader(pdf_pth)
    pages = loader.load() # loading the pages
    if pages:
        logging.info(f"{pdf_pth} loaded")
        logging.info(f"{len(pages)} pages lodaed")
    else:
        logging.info(f"{tmp_path} not loaded")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    logging.info(f"chunks:{len(chunks)} created")
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory=CHROMA_DB)

    return vectorstore

@st.cache_resource
def load_chain(_vectorstore):
    llm = OllamaLLM(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant with access to two sources:

        1. Document context from the uploaded PDF
        2. Chat history from our conversation

        Rules:
        - For questions about document content → use Document Context
        - For conversational questions (summarize our chat, what did I ask, etc.) → use Chat History
        - If not found in either → say 'This information is not found.'

        Document Context: {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
        "context":RunnableLambda(lambda x:x["question"])| _vectorstore.as_retriever(search_kwargs={'k':10})|format_docs,
        "question":RunnableLambda(lambda x:x["question"]),
        "chat_history":RunnableLambda(lambda x:x["chat_history"])}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# load resources
vectorstore  = load_vectorstore(tmp_path)
chain = load_chain(vectorstore)
if chain:
    logging.info("Chain Loaded")

# chat history

if "messages" not in st.session_state:
    st.session_state.messages= []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating=False
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


query = st.chat_input("Ask something about upload...", disabled=st.session_state.is_generating)
if query:
    # show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role":"user","content":query})
    st.session_state.chat_history.append(HumanMessage(content = query))

    # llm response
    with st.chat_message("assistant"):
        st.session_state.is_generating=True
        answer = st.write_stream(chain.stream({"question":query, "chat_history":st.session_state.get("chat_history",[])}))
        st.session_state.is_generating=False
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.chat_history.append(AIMessage(content=answer))
