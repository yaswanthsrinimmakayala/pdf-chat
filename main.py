import os
import logging
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
st.title("Chat about Transformers")
st.caption("Ask anything about the Transformer Architecture")

FILE_NAME = "170603762v7.pdf"
EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.2"
CHROMA_DB = os.path.join(os.path.dirname(__file__),"chroma_db")

@st.cache_resource
def load_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  
    if os.path.exists(CHROMA_DB):
        logging.info("Loaded existing vectorstore")
        return Chroma(persist_directory=CHROMA_DB,embedding_function=embeddings)
    # opening the PDF
    loader = PyPDFLoader(FILE_NAME)
    pages = loader.load() # loading the pages
    if pages:
        logging.info(f"{FILE_NAME} loaded")
        logging.info(f"{len(pages)} pages lodaed")
    else:
        logging.info(f"{FILE_NAME} not loaded")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    logging.info(f"chunks:{len(chunks)} created")
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory=CHROMA_DB)

    return vectorstore

@st.cache_resource
def load_chain(_vectorstore):
    llm = OllamaLLM(model=LLM_MODEL)
    prompt = PromptTemplate.from_template("""
            You are an expert assistant that answers questions strictly based on the provided context.

            Rules:
            - Answer ONLY from the context below
            - Never use outside knowledge
            - If the answer is not in the context, say exactly: "This information is not found in the document."
            - Be specific and detailed in your answers
            - If the question asks about a concept, explain it clearly using the context
            - Quote directly from the context when relevant

            Context: {context}

            Question: {question}

            Answer:
            """)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
        "context":_vectorstore.as_retriever(search_kwargs={'k':10})|format_docs,
        "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# load resources
vectorstore  = load_vectorstore()
chain = load_chain(vectorstore)
if chain:
    logging.info("Chain Loaded")

# chat history

if "messages" not in st.session_state:
    st.session_state.messages= []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

query = st.chat_input("Ask something about Transformer Architecture...")
if query:
    # show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role":"user","content":query})

    # llm response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke(query)
        st.write(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})

