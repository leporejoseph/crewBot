# src/utils/document_handler.py
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

def handle_document_upload(langchain_upload_docs_selected):
    if langchain_upload_docs_selected:
        uploaded_files = st.file_uploader("Upload documents", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
        if uploaded_files:
            documents = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join("files", uploaded_file.name)
                if not os.path.exists(file_path):
                    with st.spinner("Analyzing your document..."):
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())
                if uploaded_file.type == "application/pdf":
                    pdf_documents = PyPDFLoader(file_path).load()
                    for doc in pdf_documents:
                        doc.metadata['source'] = uploaded_file.name
                    documents.extend(pdf_documents)
                elif uploaded_file.type == "text/plain":
                    content = uploaded_file.read().decode("utf-8")
                    documents.append(Document(page_content=content, metadata={"source": uploaded_file.name}))
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                all_splits = [chunk for doc in documents for chunk in text_splitter.split_documents([doc])]
                if not st.session_state.vectorstore:
                    st.session_state.vectorstore = Chroma.from_documents(documents=all_splits, embedding=st.session_state.embedding_model)
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever()
                if 'retriever' in st.session_state and st.session_state.retriever:
                    if not st.session_state.qa_chain:
                        try:
                            system_prompt = "Use the given context to answer the question. If you don't know the answer, say you don't know. Use three sentences maximum and keep the answer concise. Context: {context}"
                            prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
                            question_answer_chain = create_stuff_documents_chain(st.session_state.llm, prompt)
                            st.session_state.qa_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)
                        except Exception as e:
                            st.error(f"Error initializing QA chain: {e}")
                            st.session_state.qa_chain = None
