# src/utils/document_handler.py

import os, re
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def handle_document_upload(langchain_upload_docs_selected):
    """Handle document upload and processing."""
    if langchain_upload_docs_selected:
        uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            documents = process_uploaded_files(uploaded_files)
            if documents:
                process_documents(documents)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return a list of documents."""
    documents = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join("files", uploaded_file.name)
        if not os.path.exists(file_path):
            save_uploaded_file(file_path, uploaded_file)
        if uploaded_file.type == "application/pdf":
            documents.extend(load_pdf_documents(file_path, uploaded_file))
        elif uploaded_file.type == "text/plain":
            documents.append(load_text_document(uploaded_file))
    return documents

def save_uploaded_file(file_path, uploaded_file):
    """Save the uploaded file to the specified path."""
    with st.spinner("Analyzing your document..."):
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

def load_pdf_documents(file_path, uploaded_file):
    """Load PDF documents from the specified file path."""
    pdf_documents = PyPDFLoader(file_path).load()
    for doc in pdf_documents:
        doc.metadata['source'] = uploaded_file.name
    return pdf_documents

def load_text_document(uploaded_file):
    """Load a text document from the uploaded file."""
    content = uploaded_file.read().decode("utf-8")
    return Document(page_content=content, metadata={"source": uploaded_file.name})

def process_documents(documents):
    """Process documents for embedding and retrieval."""
    def clean_text(text):
        """Clean text by removing special characters and extra spaces."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()

    # Clean the content of each document
    cleaned_documents = []
    for doc in documents:
        cleaned_content = clean_text(doc.page_content)
        if cleaned_content:
            cleaned_documents.append(Document(page_content=cleaned_content, metadata=doc.metadata))

    if not cleaned_documents:
        st.error("No valid content found in the uploaded documents.", icon="🚨")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_splits = [chunk for doc in cleaned_documents for chunk in text_splitter.split_documents([doc])]
    
    if not all_splits:
        st.error("Failed to split documents into valid chunks.", icon="🚨")
        return

    if not st.session_state.vectorstore:
        st.session_state.vectorstore = Chroma.from_documents(documents=all_splits, embedding=st.session_state.embedding_model)
        st.session_state.retriever = st.session_state.vectorstore.as_retriever()
    
    if 'retriever' in st.session_state and st.session_state.retriever and not st.session_state.qa_chain:
        initialize_qa_chain()

def initialize_qa_chain():
    """Initialize the QA chain for document retrieval."""
    try:
        system_prompt = "Use the given context to answer the question. If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise. Context: {context}"
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
        st.session_state.qa_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)
    except Exception as e:
        st.error(f"Error initializing QA chain: {e}", icon="🚨")
        st.session_state.qa_chain = None
