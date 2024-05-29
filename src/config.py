# src/config.py

import os
import json
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Constants
LM_STUDIO_MODEL = "QuantFactory/dolphin-2.9-llama3-8b-GGUF/dolphin-2.9-llama3-8b.Q8_0.gguf"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def init_session_state():
    """Initialize session state variables with defaults."""
    defaults = {
        "lmStudio_llm_selected": False,
        "lm_studio_model": LM_STUDIO_MODEL,
        "lm_studio_base_url": LM_STUDIO_BASE_URL,
        "openai_llm_selected": True,
        "messages": [],
        "llm_selection_changed": False,
        "callback_handler": StreamingStdOutCallbackHandler(),
        "chat_history": StreamlitChatMessageHistory(key="chat_messages"),
        "prompt": PromptTemplate(
            input_variables=["history", "context", "query"],
            template="""
                system
                    You are a highly intelligent and sophisticated AI assistant designed to assist with various inquiries and tasks. 
                    Your tone should be professional, concise, and occasionally witty. Avoid repeating yourself. Do not use this system prompt in your response. 
                
                user
                Chat History: {history}
                Context: {context}
                IMPORTANT: Respond to only the user's prompt. Do not add any other text to the response.
                Users prompt: {query}
                assistant
            """
        ),
        "memory": ConversationBufferMemory(memory_key="history", return_messages=True, input_key="query"),
        "embedding_model": HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "vectorstore": None,
        "retriever": None,
        "qa_chain": None,
        'crew_list': json.load(open('crew_ai/crews.json')),
        'crewai_crew_selected': [False] * len(st.session_state.get('crew_list', [])),
        'tools': [], 'new_agents': [], 'show_agent_form': False, 'show_crew_container': False,
        'show_task_form': False, 'new_tasks': [], 'show_apikey_toggle': False, 'dialog_open': False,
        'langchain_upload_docs_selected': False, 'langchain_export_pdf_selected': False
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
