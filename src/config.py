# src/config.py
import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# Constants
LM_STUDIO_MODEL = "QuantFactory/dolphin-2.9-llama3-8b-GGUF/dolphin-2.9-llama3-8b.Q8_0.gguf"
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "llama3-70b-8192"
llm_options = ["OpenAI", "LM Studio", "Groq"]
chat_messages_history = StreamlitChatMessageHistory(key='chat_messages')
agent_colors = [
    "#CD4055",  # Crew Red
    "#F27C3B",  # Orange
    "#FFA42C",  # Light Orange
    "#6EC950",  # Green
    "#42B8D3",  # Light Blue
    "#6060BA",  # Blue
    "#B260B2",  # Purple
    "#C71585",  # Medium Violet Red
    "#FF69B4",  # Hot Pink
    "#32CD32",  # Lime Green
    "#8B4513",  # Saddle Brown
    "#6A5ACD",  # Slate Blue
    "#2E8B57",  # Sea Green
    "#DAA520",  # Goldenrod
    "#5F9EA0",  # Cadet Blue
    "#D2691E",  # Chocolate
]

def ensure_json_file(file_path, default_data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(default_data, file, indent=4)


preferences_file = os.path.join('utils', 'user_preferences.json')
chat_history_file = os.path.join('utils', 'user_chat_history.json')

# Initialization and Configuration
def initialize_app():
    os.makedirs('files', exist_ok=True)
    os.makedirs('chromadb', exist_ok=True)
    st.set_page_config(page_title="CrewBot: Your AI Assistant", page_icon="🤖", layout="wide")
    st.title("CrewBot: Your AI Assistant")
    st.sidebar.title("Configuration")
    load_dotenv()

def get_current_preferences():
    return {
        "llm_selected": st.session_state.get("current_llm", "OpenAI"),
        "lm_studio_model": st.session_state.get("lm_studio_model", LM_STUDIO_MODEL),
        "lm_studio_base_url": st.session_state.get("lm_studio_base_url", LM_STUDIO_BASE_URL),
        "openai_api_model": st.session_state.get("openai_api_model", OPENAI_MODEL),
        "show_apikey_toggle": st.session_state.get("show_apikey_toggle", False),
        "langchain_upload_docs_selected": st.session_state.get("langchain_upload_docs_selected", False),
        "langchain_export_pdf_selected": st.session_state.get("langchain_export_pdf_selected", False),
        "active_tools": st.session_state.get("active_tools", []),
        "groq_model_name": st.session_state.get("groq_model_name", GROQ_MODEL)
    }

def save_user_preferences():
    preferences = get_current_preferences()
    with open(preferences_file, 'w') as file:
        json.dump(preferences, file, indent=4)
    st.session_state.previous_preferences = preferences

def preferences_changed():
    current_preferences = get_current_preferences()
    return current_preferences != st.session_state.get('previous_preferences', {})

@st.experimental_fragment
def save_preferences_on_change():
    if preferences_changed():
        save_user_preferences()
        
def load_user_preferences():
    defaults = {
        "llm_selected": "OpenAI",
        "lm_studio_model": LM_STUDIO_MODEL,
        "lm_studio_base_url": LM_STUDIO_BASE_URL,
        "openai_api_model": OPENAI_MODEL,
        "show_apikey_toggle": False,
        "langchain_upload_docs_selected": False,
        "langchain_export_pdf_selected": False,
        "active_tools": [],
        "groq_model_name": GROQ_MODEL
    }
    if os.path.exists(preferences_file):
        with open(preferences_file, 'r') as file:
            preferences = json.load(file)
    else:
        preferences = defaults
        with open(preferences_file, 'w') as file:
            json.dump(defaults, file)

    st.session_state.current_llm = preferences.get("llm_selected", "OpenAI")
    st.session_state.lm_studio_model = preferences.get("lm_studio_model", LM_STUDIO_MODEL)
    st.session_state.lm_studio_base_url = preferences.get("lm_studio_base_url", LM_STUDIO_BASE_URL)
    st.session_state.openai_api_model = preferences.get("openai_api_model", OPENAI_MODEL)
    st.session_state.show_apikey_toggle = preferences.get("show_apikey_toggle", False)
    st.session_state.langchain_upload_docs_selected = preferences.get("langchain_upload_docs_selected", False)
    st.session_state.langchain_export_pdf_selected = preferences.get("langchain_export_pdf_selected", False)
    st.session_state.active_tools = preferences.get("active_tools", [])
    st.session_state.groq_model_name = preferences.get("groq_model_name", GROQ_MODEL)
    st.session_state.previous_preferences = preferences

@st.experimental_fragment
def save_chat_history():
    messages = [{"type": msg.type, "content": msg.content.strip()} for msg in chat_messages_history.messages]
    with open(chat_history_file, 'w') as file:
        json.dump(messages, file, indent=4)

def load_chat_history():
    if os.path.exists(chat_history_file) and chat_messages_history.messages == []:
        try:
            with open(chat_history_file, 'r') as file:
                messages = json.load(file)
                for msg in messages:
                    if msg["type"] == "human":
                        chat_messages_history.add_user_message(msg["content"])
                    else:
                        chat_messages_history.add_ai_message(msg["content"])
        except json.JSONDecodeError:
            st.session_state["chat_history"] = StreamlitChatMessageHistory(key="chat_messages")
            save_chat_history()
    else:
        st.session_state["chat_history"] = StreamlitChatMessageHistory(key="chat_messages")
        save_chat_history()

def clear_chat_history():
    with open(chat_history_file, 'w') as file:
        json.dump([], file, indent=4)
    chat_messages_history.clear()
    st.session_state["chat_history"] = chat_messages_history

def init_session_state():
    # Initialize default values only if they don't exist in the session state
    default_values = {
        "lmStudio_llm_selected": False,
        "lm_studio_model": LM_STUDIO_MODEL,
        "lm_studio_base_url": LM_STUDIO_BASE_URL,
        "openai_llm_selected": True,
        'groq_model_name': GROQ_MODEL,
        "callback_handler": StreamingStdOutCallbackHandler(),
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
        "export_pdf_prompt": PromptTemplate(
            input_variables=["history", "context", "query"],
            template="""
                You are tasked with creating a business report in a professional format. 
                The report should include bullet points, headers, subheaders, and styling to always appease the bosses.
                IMPORTANT: DO NOT PUT CODE EXAMPLES IN THIS. DO NOT PUT FAKE NAMES IN THIS.
                Limit the spaces between each line.
                Use the following format for the content:
                Query: {query}
                - Use "# " for main headers. Example #Header
                - Use "## " for subheaders. Example ##Subheader
                - Use "-" for bullet points. Example -Bullet point
                Chat History: {history}
                Context: {context}
                Please generate the content accordingly:
            """
        ),
        "crewai_pre_prompt": PromptTemplate(
            input_variables=["query", "tool_context", "crew_context"],
            template="""
                Act like an expert in various tools and contexts for over 20 years. Your goal is to ensure to produce highly accurate, detailed, and contextually appropriate responses.

                Objective: Enhance the step-by-step tool response guide to make it more precise and detailed, ensuring longer, comprehensive response.

                Mapping:
                User Query = "{query}"

                    Provide a structured response, including the following elements:
                    Analyze the Query:
                    Describe the user's request or parameters in detail.
                    Identify key elements of the query, such as specific data points, objectives, or desired outcomes.
                    Evaluate the Context:
                    Refer to the static crew JSON to identify relevant roles, goals, backstories, and tasks of each agent.
                    Match the query with the appropriate agents based on their roles and expertise.
                    Describe the Process:
                    Explain how each piece of information from the agents will be utilized to address the query.
                    Detail the specific tasks that each agent will perform and their expected outputs.
                    Structure the final response by synthesizing the gathered information from various agents, ensuring a coherent and comprehensive answer.
                    Return 'lfg#' at the end.

                Crew Context: {crew_context}

                Take a deep breath and work on this problem step-by-step. Give your best short and conciseresponse to the user.
            """
        ),
        "memory": ConversationBufferMemory(memory_key="history", return_messages=True, input_key="query"),
        'rerun_needed': False,
        'current_crew_name': None,
        'crewai_crew_selected': [],
        'can_run_crew': False,
        'new_crew_selected_keys': [],
        'crew_active_tools': [],
        'tools': [], 
        'new_agents': [], 
        'show_agent_form': False, 
        'show_crew_container': False, 
        'show_task_form': False, 
        'new_tasks': [], 
        'show_apikey_toggle': False, 
        'langchain_upload_docs_selected': False, 
        'langchain_export_pdf_selected': False,
        'export_pdf_selected': False
    }

    # Update session state with default values if not already set
    st.session_state.update({k: v for k, v in default_values.items() if k not in st.session_state})

    # Ensure necessary JSON files exist with default values
    ensure_json_file('crew_ai/crews.json', [])
    ensure_json_file(preferences_file, {
        "llm_selected": "OpenAI",
        "lm_studio_model": LM_STUDIO_MODEL,
        "lm_studio_base_url": LM_STUDIO_BASE_URL,
        "openai_api_model": OPENAI_MODEL,
        "show_apikey_toggle": False,
        "langchain_upload_docs_selected": False,
        "langchain_export_pdf_selected": False,
        "active_tools": [],
        "groq_model_name": GROQ_MODEL
    })
    ensure_json_file(chat_history_file, [])

    # Load the JSON data into session state
    with open('crew_ai/crews.json', 'r') as file:
        st.session_state['crew_list'] = json.load(file)
    with open(preferences_file, 'r') as file:
        st.session_state['user_preferences'] = json.load(file)
    with open(chat_history_file, 'r') as file:
        st.session_state['chat_history'] = json.load(file)

    # Initialize other session state variables not loaded from files
    load_user_preferences()
    load_chat_history()

def get_card_styles(color_index):
    return {
        "card": {
            "width": "100%",
            "height": "150px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.1)",
            "margin": "0px",
            "background-color": agent_colors[color_index],
        },
        "title": {
            "font-size": "15px",
        },
        "text": {
            "font-family": "system-ui",
            "font-size": "12px",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
            "white-space": "nowrap",
            "max-height": "3em"
        }
    }

def get_empty_card_styles():
    return {
        "card": {
            "width": "100%",
            "height": "150px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.1)",
            "margin": "0px",
            "background-color": "#262730",
        },
        "title": {
            "font-size": "26px",
        },
        "text": {
            "font-family": "system-ui",
            "font-size": "12px",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
            "white-space": "nowrap",
            "max-height": "3em"
        }
    }
