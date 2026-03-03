import os
import json
import hashlib
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
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

EMBED_MODEL = "mxbai-embed-large"
LLM_MODEL = "llama3.1:8b"
CHROMA_DB = f"./chroma_db/{file_hash}"

@st.cache_resource
def load_vectorstore(file_hash,pdf_pth):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)  
    if os.path.exists(CHROMA_DB) and os.listdir(CHROMA_DB):
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    logging.info(f"chunks:{len(chunks)} created")
    
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory=CHROMA_DB)
    vectorstore.persist()
    return vectorstore

@st.cache_resource
def load_chain(_vectorstore):
    llm = OllamaLLM(model=LLM_MODEL,temperature=0)
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a document-grounded assistant.

        You have access to:
        1. Retrieved context from a PDF (with page numbers)
        2. Chat history (only to understand follow-up questions)

        -----------------------
        STRICT RULES
        -----------------------

        1. Answer ONLY using the provided Document Context.
        2. If the answer is not explicitly found in the context, respond exactly:
        "This information is not found."

        3. Every factual statement MUST include a citation.
        4. Citations MUST use this exact format:
        [Page X]

        5. If multiple pages support a statement, list them like:
        [Page 2, Page 5]

        6. Do NOT invent page numbers.
        7. Do NOT use knowledge outside the document.
        8. Do NOT mention the chat history unless necessary to resolve a follow-up.

        -----------------------
        ANSWER FORMAT
        -----------------------

        - Be concise.
        - Use short paragraphs.
        - Place citations immediately after the sentence they support.
        - Do NOT place citations at the end of the entire answer.
        - Do NOT include a references section.

        -----------------------
        Document Context:
        {context}
        """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    def format_docs(docs):
        results = []
        for doc in docs:
            page = doc.metadata.get('page','unknown')
            results.append(f"[Page {page}] \n {doc.page_content}")
        return "\n\n".join(results)

    chain = (
        {
        "context":RunnableLambda(lambda x:x["question"])| _vectorstore.as_retriever( search_type="mmr",search_kwargs={'k':5})|format_docs,
        "question":RunnableLambda(lambda x:x["question"]),
        "chat_history":RunnableLambda(lambda x:x["chat_history"])}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# load resources
vectorstore  = load_vectorstore(file_hash,tmp_path)
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

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
query = st.chat_input("Ask something about upload...", disabled=st.session_state.is_generating)
if query:
    # show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role":"user","content":query})
    st.session_state.chat_history.append(HumanMessage(content = query))
    answer = ""
    # llm response
    with st.chat_message("assistant"):
        st.session_state.is_generating=True
        message_placeholder = st.empty() 
        for chunk in chain.stream({"question":query, "chat_history":st.session_state.get("chat_history",[])}):
            answer+=chunk
            message_placeholder.markdown(answer) 
        st.session_state.is_generating=False
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.session_state.chat_history.append(AIMessage(content=answer))
