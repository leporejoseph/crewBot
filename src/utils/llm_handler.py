# src/utils/llm_handler.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os, asyncio, functools

@functools.lru_cache(maxsize=None)
def init_llm(api_key, model_name, base_url):
    """Initialize the LLM based on the selected model and API key."""
    callback_handler = st.session_state.callback_handler
    return ChatOpenAI(model=model_name, base_url=base_url, temperature=0.75, api_key=api_key if model_name.startswith("gpt-") else "NA", streaming=True, callbacks=[callback_handler])

def set_initial_llm():
    """Set the initial LLM state based on the selected model."""
    if st.session_state.lmStudio_llm_selected:
        st.session_state.llm = init_llm(None, st.session_state.lm_studio_model, st.session_state.lm_studio_base_url)
        st.session_state.llm_name = "LM Studio"
        st.session_state.llm_model = st.session_state.lm_studio_base_url
    else:
        st.session_state.llm = init_llm(os.getenv("OPENAI_API_KEY"), st.session_state.get('openai_api_model', 'default-model'), None)
        st.session_state.llm_name = "OpenAI"
        st.session_state.llm_model = st.session_state.get('openai_api_model', 'Default')

def update_api_key():
    """Update the API key and reinitialize the LLM."""
    new_key = st.session_state.get('openai_api_key')
    os.environ["OPENAI_API_KEY"] = new_key
    with open(".env", "w") as env_file:
        env_file.write(f"OPENAI_API_KEY={new_key}\n")
    set_initial_llm()

def toggle_selection(key):
    """Toggle the selection between LM Studio and OpenAI."""
    st.session_state.lmStudio_llm_selected = key == "lmStudio_llm_selected"
    st.session_state.openai_llm_selected = key == "openai_llm_selected"
    set_initial_llm()

async def get_response_async(llm, user_query, tool, chat_messages_history, context=""):
    try:
        qa_chain = st.session_state.get('qa_chain')
        if tool == "upload_documents" and qa_chain:
            response = qa_chain({"query": user_query, "context": context})
            return response.get('answer', "No documents found, please upload documents.")
        else:
            prompt = st.session_state['prompt']
            parser = StrOutputParser()
            chain = prompt | llm | parser
            response_chunks = [chunk async for chunk in chain.astream({"query": user_query, "context": context, "history": chat_messages_history})]
            return "".join(response_chunks)
    except Exception as e:
        st.error(f"Error in get_response_async: {e}")
        return ""

def get_response(llm, user_query, tool, chat_messages_history, context=""):
    return asyncio.run(get_response_async(llm, user_query, tool, chat_messages_history, context))