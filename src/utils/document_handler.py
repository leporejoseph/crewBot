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
from fpdf import FPDF

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

def download_pdf(formatted_pdf_text: str, crew_name: str):
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "Summary Report", 0, 1, "C")

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def chapter_title(self, title):
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, title, 0, 1, "L")
            self.ln(4)

        def chapter_body(self, body):
            self.set_font("Arial", size=12)
            self.multi_cell(0, 10, body)
            self.ln()

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    last_was_header = False
    for line in formatted_pdf_text.split("\n"):
        if line.startswith("# "):  # Header 1
            pdf.chapter_title(line[2:])
            last_was_header = True
        elif line.startswith("## "):  # Header 2
            pdf.set_font("Arial", "B", 10)
            pdf.cell(0, 10, line[3:], 0, 1, "L")
            last_was_header = True
        elif line.startswith("- "):  # Bullet point
            pdf.set_font("Arial", size=12)
            pdf.cell(5, 5, "-", 0, 0)
            pdf.multi_cell(0, 10, line[2:])
            last_was_header = False
        else:
            if last_was_header:
                pdf.ln(2)
            pdf.chapter_body(line)
            last_was_header = False

    # Define the path to save the PDF
    output_dir = os.path.join("files")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    file_name = os.path.join(output_dir, f"{crew_name}_Summary.pdf")
    
    # Save the PDF
    pdf.output(file_name)
    
    # Provide a download button for the saved PDF
    st.download_button(
        label=f"Download {crew_name}_Summary.pdf",
        data=open(file_name, "rb").read(),
        file_name=f"{crew_name}_Summary.pdf",
        mime="application/pdf"
    )
