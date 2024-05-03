import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from crewai_crews.businessreqs_crew import create_crewai_setup
from utilities import StreamToExpander, get_response

import os
import sys
import time

# Set up Streamlit configurations
st.set_page_config(
    page_title="CrewBot: Your AI Assistant",
    page_icon="🤖",
    layout="wide", 
)
st.title("CrewBot: Your AI Assistant")
st.sidebar.title("Configuration")

# Load environment variables
load_dotenv()

# Fetch the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Global variables for storing task outputs
lm_studio_model = "QuantFactory/dolphin-2.9-llama3-8b-GGUF/dolphin-2.9-llama3-8b.Q8_0.gguf"
lm_studio_base_url = "http://localhost:1234/v1"
task_values = []
agent_task_outputs = []
first_load_settings = True

# Function to initialize the LLM based on selection
def init_llm(api_key, model_name, base_url):
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key if model_name == "gpt-3.5-turbo" else "NA",
        streaming=True,
    )

# Initialize session state variables if not present
def init_session_state():
    st.session_state.setdefault("lmStudio_llm_selected", False)
    st.session_state.setdefault("lm_studio_model", lm_studio_model)
    st.session_state.setdefault("lm_studio_base_url", lm_studio_base_url)
    st.session_state.setdefault("openai_llm_selected", bool(openai_api_key))
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("chat_history", [AIMessage(content="Hello, I am a bot. How can I help you?")])


# Set initial LLM state based on "LM Studio" by default
def set_initial_llm():
    global llm  # Ensure `llm` is global here first

    if st.session_state.lmStudio_llm_selected:
        llm = init_llm(
            None, 
            st.session_state.lm_studio_model, 
            st.session_state.lm_studio_base_url,
        )
    else:
        llm = init_llm(openai_api_key, "gpt-3.5-turbo", None)

# Callback function to update the API key and save it to the .env file
def update_api_key():
    global llm  # Ensure `llm` is global here first

    new_key = st.session_state.get('openai_api_key')

    # Update the environment variable
    os.environ["OPENAI_API_KEY"] = new_key

    # Rewrite the .env file with the new key
    with open(".env", "w") as env_file:
        env_file.write(f"OPENAI_API_KEY={new_key}\n")

    # Reinitialize the LLM
    llm = init_llm(new_key, "gpt-3.5-turbo", None)

# Callback function to update LLM and Streamlit state
def toggle_selection(key):
    global llm  # Ensure `llm` is global here first

    if key == "lmStudio_llm_selected":
        st.session_state.openai_llm_selected = False
        llm = init_llm(
            None, 
            st.session_state.lm_studio_model, 
            st.session_state.lm_studio_base_url,
        )
    elif key == "openai_llm_selected":
        st.session_state.lmStudio_llm_selected = False
        llm = init_llm(openai_api_key, "gpt-3.5-turbo", None)

# Initialize session state and set initial LLM
init_session_state()
set_initial_llm()

# CSS styling block
st.html("""
    <html>
        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                }
                h2 {
                    color: #CD4055; /* Custom color for header, CrewAI inspired */
                }
            </style>
        </head>
    </html>
""")

with st.sidebar.expander("**LLM Selection**", True):
    st.checkbox("LM Studio", key="lmStudio_llm_selected", on_change=toggle_selection, args=("lmStudio_llm_selected",))
    st.checkbox("OpenAI", key="openai_llm_selected", on_change=toggle_selection, args=("openai_llm_selected",))

    if st.session_state.lmStudio_llm_selected:
        st.text_input(
            'LM Studio Model',
            st.session_state.lm_studio_model,
            on_change=lambda: set_initial_llm(),
            key='lm_studio_model',
        )
        st.text_input(
            'LM Studio Base URL',
            st.session_state.lm_studio_base_url,
            on_change=lambda: set_initial_llm(),
            key='lm_studio_base_url',
        )
    elif st.session_state.openai_llm_selected:
        st.text_input(
            'OpenAI API Key',
            openai_api_key,
            on_change=update_api_key,
            key='openai_api_key',
            type='password'
        )

# Crew section with HTML
with st.sidebar.expander("***crew***:red[***ai***]", True):
    crewai_bus_reqs_selected = st.toggle("Run Crew: Create Business Requirements", value=False)

    st.html("""
        <html>
            <head>
                <style>
                    .crewai-description {
                        font-size: 11px;
                        font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif;
                    }
                    .crewai-ul {
                        font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif;
                        list-style-type: disc;
                        padding-left: 20px;
                    }
                    .crewai-li {
                        margin-bottom: 10px;
                        font-size: 11px !important;
                    }
                    .crewai-img {
                        width: 69px;
                        margin-bottom: 10px;
                    }
                </style>
            </head>
        </html>
        """)

    st.markdown("""
        <html>
            <body class='crewai-body'>
                <p class='crewai-description'>Enter information about your web app, then sit back and watch as a crew conducts research, covering:</p>
                <ul class='crewai-ul'>
                    <li class='crewai-li'>Market Research Analyst</li>
                    <li class='crewai-li'>Technology Expert</li>
                    <li class='crewai-li'>Business Development Consultant</li>
                    <li class='crewai-li'>Project Manager</li>
                    <li class='crewai-li'>Summary</li>
                </ul>
                <a href='https://www.crewai.com/' target='_blank' class='crewai-link'>
                    <img class='crewai-img' src="https://www.crewai.com/assets/crew_only-ce3e8e1afde0977caeaa861aab72f1cfee3c88a79127d6e2bea8d9b2066f5eb1.png" alt="CrewAI Logo">
                </a>
            </body>
        </html>
        """, True)

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# Input field for user's current question
user_query = st.chat_input("What's on your mind?")
if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if crewai_bus_reqs_selected:
            start_time = time.time()
            with st.spinner("Agents Working..."):
                with st.expander("Crew Interaction Logs", False):
                    sys.stdout = StreamToExpander(st, agent_task_outputs, task_values)
                    response = create_crewai_setup(user_query, st.session_state.chat_history, llm)

            stopwatch_placeholder = st.empty()
            end_time = time.time()
            total_time_elapsed = end_time - start_time
            stopwatch_placeholder.text(f"Total Time Elapsed: {total_time_elapsed:.2f} seconds")

            st.header("Tasks:")
            st.table({"Tasks": task_values})

            st.header("Results:")
            st.markdown(response)
            st.session_state.messages.append({"role": "AI", "content": response})

        else:
            response = st.write_stream(get_response(llm, user_query, st.session_state.chat_history))
            st.session_state.messages.append({"role": "AI", "content": response})

    # Append response to chat history
    st.session_state.chat_history.append(AIMessage(content=response))
