# src/utils/llm_handler.py

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import asyncio
from config import GROQ_MODEL, OPENAI_MODEL, LM_STUDIO_MODEL, LM_STUDIO_BASE_URL
from utils.document_handler import download_pdf

def init_llm(api_key, model_name, llm_name, base_url):
    """Initialize the LLM based on the selected LLM name and API key."""
    callback_handler = st.session_state.callback_handler
    if llm_name == "OpenAI":
        return ChatOpenAI(model=model_name, base_url=base_url, temperature=0.75, api_key=api_key, streaming=True, callbacks=[callback_handler])
    elif llm_name == "Groq":
        return ChatGroq(model_name=model_name, temperature=0.75, groq_api_key=api_key, callbacks=[callback_handler])
    else:
        return ChatOpenAI(model=model_name, base_url=base_url, temperature=0.75, api_key=api_key, streaming=True, callbacks=[callback_handler])

def set_initial_llm():
    """Set the initial LLM state based on the selected model."""
    llm_name = st.session_state.current_llm
    if llm_name == "OpenAI":
        st.session_state.llm = init_llm(os.getenv("OPENAI_API_KEY", "N/A"), st.session_state.get('openai_api_model', OPENAI_MODEL), llm_name, None)
    elif llm_name == "LM Studio":
        st.session_state.llm = init_llm("N/A", st.session_state.get('lm_studio_model', LM_STUDIO_MODEL), llm_name, LM_STUDIO_BASE_URL)
    elif llm_name == "Groq":
        st.session_state.llm = init_llm(os.getenv("GROQ_API_KEY", "N/A"), st.session_state.get('groq_model_name', GROQ_MODEL), llm_name, None)

def update_api_key():
    """Update the API key and reinitialize the LLM."""
    env_path = ".env"
    if st.session_state.openai_llm_selected:
        new_key = st.session_state.get('openai_api_key', '')
        os.environ["OPENAI_API_KEY"] = new_key
        update_env_file(env_path, "OPENAI_API_KEY", new_key)
    elif st.session_state.groq_llm_selected:
        new_key = st.session_state.get('groq_api_key', '')
        os.environ["GROQ_API_KEY"] = new_key
        update_env_file(env_path, "GROQ_API_KEY", new_key)
    set_initial_llm()

def update_env_file(env_path, key, value):
    """Update the .env file with the given key-value pair."""
    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            lines = file.readlines()
    else:
        lines = []

    key_found = False
    new_lines = []
    for line in lines:
        if line.startswith(f'{key}='):
            new_lines.append(f'{key}={value}\n')
            key_found = True
        else:
            new_lines.append(line)
    if not key_found:
        new_lines.append(f'{key}={value}\n')

    with open(env_path, 'w') as file:
        file.writelines(new_lines)

@st.experimental_fragment
def toggle_selection(key):
    """Toggle the selection between different LLMs."""
    st.session_state.openai_llm_selected = key == "openai_llm_selected"
    st.session_state.lmStudio_llm_selected = key == "lmStudio_llm_selected"
    st.session_state.groq_llm_selected = key == "groq_llm_selected"
    set_initial_llm()

async def get_response_async(llm, user_query, tool, chat_messages_history, context=""):
    try:
        if llm is None:
            raise ValueError("LLM is not initialized.")

        tools_used = [tool for tool in st.session_state.crew_active_tools]

        if any(st.session_state.crewai_crew_selected):
            prompt = st.session_state.get('crewai_pre_prompt', st.session_state['prompt'])
            tool_context = ", ".join(tools_used)
            input_data = {"query": user_query, "crew_context": context, "tool_context": tool_context}
        elif tool == "export_pdf":
            prompt = st.session_state.get('export_pdf_prompt', st.session_state['prompt'])
            input_data = {"query": user_query, "context": context, "history": chat_messages_history}
        else:
            prompt = st.session_state['prompt']
            input_data = {"query": user_query, "context": context, "history": chat_messages_history}

        parser = StrOutputParser()
        chain = prompt | llm | parser

        with st.chat_message("assistant"):
            first_output_placeholder = st.empty()
            first_output = ""

            async for chunk in chain.astream(input_data):
                first_output += chunk
                first_output_placeholder.write(first_output)

            if tool == "export_pdf":
                with st.spinner(f"Generating PDF..."):
                    pdf_output = ""
                    async for chunk in chain.astream(input_data):
                        pdf_output += chunk

                    crew_name = "Crew"  # Default crew name if not found
                    if st.session_state.get("current_crew_name"):
                        crew_name = st.session_state.current_crew_name

                    download_pdf(pdf_output, crew_name)

            if any(st.session_state.crewai_crew_selected):
                if "lfg#" in first_output.lower():
                    st.session_state.update({"can_run_crew": True})
                else:
                    st.session_state.update({"can_run_crew": False})

            chat_messages_history.add_ai_message(first_output)

    except Exception as e:
        st.error(f"Error in get_response_async: {e}")
        return ""

def get_response(llm, user_query, tool, chat_messages_history, context=""):
    return asyncio.run(get_response_async(llm, user_query, tool, chat_messages_history, context))
