# src/app.py
import streamlit as st
from dotenv import load_dotenv
from utils.llm_handler import set_initial_llm, update_api_key, toggle_selection #, get_response
from utils.document_handler import handle_document_upload
from utils.streamlit_expander import StreamToExpander
from crewai_crews.businessreqs_crew import example_crew_data, DynamicCrewHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
import os
import sys
import time
from config import init_session_state
import json
import asyncio
import pandas as pd

# Ensure necessary directories exist
os.makedirs('files', exist_ok=True)
os.makedirs('chromadb', exist_ok=True)

# Set up Streamlit configurations
st.set_page_config(page_title="CrewBot: Your AI Assistant", page_icon="🤖", layout="wide")
st.title("CrewBot: Your AI Assistant")
st.sidebar.image(os.path.join(os.path.dirname(__file__), 'crewai_crews', 'crewai.png'), width=300)
st.sidebar.title("Configuration")
load_dotenv()

# Initialize session state and LLM
init_session_state()
set_initial_llm()

# Global variables
TOOLS = [
    {"name": "CSVSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool designed for searching within CSV files, tailored to handle structured data."},
    {"name": "CodeDocsSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool optimized for searching through code documentation and related technical documents."},
    {"name": "DOCXSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool aimed at searching within DOCX documents, ideal for processing Word files."},
    {"name": "DirectoryReadTool", "needsApiKey": False, "source": "crewai", "description": "Facilitates reading and processing of directory structures and their contents."},
    {"name": "DirectorySearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool for searching within directories, useful for navigating through file systems."},
    {"name": "FileReadTool", "needsApiKey": False, "source": "crewai", "description": "Enables reading and extracting data from files, supporting various file formats."},
    {"name": "GithubSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool for searching within GitHub repositories, useful for code and documentation search."},
    {"name": "JSONSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool designed for searching within JSON files, catering to structured data handling."},
    {"name": "MDXSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool tailored for searching within Markdown (MDX) files, useful for documentation."},
    {"name": "PDFSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool aimed at searching within PDF documents, ideal for processing scanned documents."},
    {"name": "PGSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool optimized for searching within PostgreSQL databases, suitable for database queries."},
    {"name": "RagTool", "needsApiKey": False, "source": "crewai", "description": "A general-purpose RAG tool capable of handling various data sources and types."},
    {"name": "ScrapeElementFromWebsiteTool", "needsApiKey": False, "source": "crewai", "description": "Enables scraping specific elements from websites, useful for targeted data extraction."},
    {"name": "ScrapeWebsiteTool", "needsApiKey": False, "source": "crewai", "description": "Facilitates scraping entire websites, ideal for comprehensive data collection."},
    {"name": "SerperDevTool", "needsApiKey": True, "source": "crewai", "description": "A specialized tool for development purposes, with specific functionalities under development."},
    {"name": "TXTSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool focused on searching within text (.txt) files, suitable for unstructured data."},
    {"name": "WebsiteSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool for searching website content, optimized for web data extraction."},
    {"name": "XMLSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool designed for searching within XML files, suitable for structured data formats."},
    {"name": "YoutubeChannelSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool for searching within YouTube channels, useful for video content analysis."},
    {"name": "YoutubeVideoSearchTool", "needsApiKey": False, "source": "crewai", "description": "A RAG tool aimed at searching within YouTube videos, ideal for video data extraction."}
]

TOOL_NAMES = [tool["name"] for tool in TOOLS]

# Persistent global variables
if 'crew_list' not in st.session_state:
    file_path = 'crewai_crews/crews.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            st.session_state.crew_list = json.load(file)
    else:
        st.session_state.crew_list = [example_crew_data]
        with open(file_path, 'w') as file:
            json.dump(st.session_state.crew_list, file, indent=4)
if 'crewai_crew_selected' not in st.session_state:
    st.session_state.crewai_crew_selected = [False] * len(st.session_state.crew_list)
else:
    if len(st.session_state.crewai_crew_selected) != len(st.session_state.crew_list):
        st.session_state.crewai_crew_selected = [False] * len(st.session_state.crew_list)

if 'tools' not in st.session_state:
    st.session_state.tools = []
if 'new_agents' not in st.session_state:
    st.session_state.new_agents = []
if 'show_agent_form' not in st.session_state:
    st.session_state.show_agent_form = False
if 'show_crew_container' not in st.session_state:
    st.session_state.show_crew_container = False
if 'show_task_form' not in st.session_state:
    st.session_state.show_task_form = False
if 'new_tasks' not in st.session_state:
    st.session_state.new_tasks = []
if 'show_apikey_toggle' not in st.session_state:
    st.session_state.show_apikey_toggle = False

# Initialize variables
agent_task_outputs = []

chat_messages_history = StreamlitChatMessageHistory(key='chat_messages')

# CSS styling block
st.markdown("""<style>body { font-family: Arial, sans-serif; } h2 { color: #CD4055; }</style>""", unsafe_allow_html=True)

# Sidebar configuration for LLM Selection
with st.sidebar.expander("**LLM Selection**", True):
    llm_options = ["OpenAI", "LM Studio"]
    llm_selected = st.selectbox("LLM", llm_options, index=0)

    st.session_state.openai_llm_selected = (llm_selected == "OpenAI")
    st.session_state.lmStudio_llm_selected = (llm_selected == "LM Studio")
    toggle_selection("openai_llm_selected" if st.session_state.openai_llm_selected else "lmStudio_llm_selected")

    if st.session_state.openai_llm_selected:
        st.session_state.current_llm = f"OpenAI: {st.session_state.get('openai_api_model', 'Default')}"
        st.text_input('Model', st.session_state.openai_api_model, on_change=set_initial_llm, key='openai_api_model')

    elif st.session_state.lmStudio_llm_selected:
        st.session_state.current_llm = f"LM Studio: {st.session_state.get('lm_studio_base_url', 'Default')}"
        st.text_input('Model', st.session_state.lm_studio_model, on_change=set_initial_llm, key='lm_studio_model')
        st.text_input('Base URL', st.session_state.lm_studio_base_url, on_change=set_initial_llm, key='lm_studio_base_url')

    if st.session_state.openai_llm_selected:
        st.session_state.show_apikey_toggle = st.toggle("Show Api Key", value=False, key='show_openai_key')
        open_ai_apikey = st.empty()

        if st.session_state.show_apikey_toggle:
            open_ai_apikey.text_input(
                'OpenAI API Key',
                os.getenv("OPENAI_API_KEY"),
                on_change=update_api_key,
                key='openai_api_key',
                type='default'
            )
        else:
            open_ai_apikey.empty()

# Sidebar configuration for LangChain Tools
with st.sidebar.expander("LangChain Tools :parrot: :link:", True):
    langchain_upload_docs_selected = st.toggle("Upload Documents", value=True)
    langchain_export_pdf_selected = st.toggle("Custom Tool: Export PDF", value=False)

# Function to reset form states
def reset_form_states():
    st.session_state.show_crew_container = False
    st.session_state.show_agent_form = False
    st.session_state.show_task_form = False

# Function to update agent list display
def update_agent_list(container):
    with container:
        if not st.session_state.new_agents:
            st.warning("No Agents Added")
        else:
            active_tools = st.session_state.get("active_tools", [])
            agent_list_items = "".join(
                f"<div style='margin-bottom: 10px;'><strong>{agent['role']}</strong><br>"
                f"<div style='margin-left: 20px;'><strong>Goal:</strong> {agent['goal']}<br>"
                f"<strong>Backstory:</strong> {agent['backstory']}<br>"
                f"<strong>LLM:</strong> {agent.get('llm', st.session_state.current_llm)}<br>"
                f"<strong>Allow Delegation:</strong> {agent.get('allow_delegation', False)}<br>"
                f"<strong>Memory:</strong> {agent.get('memory', True)}<br>"
                f"<strong>Tools:</strong> {list(filter(lambda tool: tool in active_tools, agent.get('tools', [])))}</div></div>"
                for agent in st.session_state.new_agents
            )
            st.markdown(f"""
            <style>.crewai-description {{ font-size: 11px; font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; padding-left: 20px; }}
                .crewai-ul {{ font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; list-style-type: disc; padding-left: 20px; }}
                .crewai-li {{ margin-bottom: 10px; font-size: 11px !important; }}
                .crewai-img {{ width: 69px; margin-bottom: 10px; }}</style>
            {agent_list_items}
            """, unsafe_allow_html=True)

# Function to update task list display
def update_task_list(container):
    with container:
        if not st.session_state.new_tasks:
            st.warning("No Tasks Added")
        else:
            active_tools = st.session_state.get("active_tools", [])
            task_list_items = "".join(
                f"<div style='margin-bottom: 10px;'><strong>{task['description']}</strong><br>"
                f"<div style='margin-left: 20px;'>"
                f"<strong>Agent:</strong> {st.session_state.new_agents[task['agent_index']]['role']}<br>"
                f"<strong>Expected Output:</strong> {task['expected_output']}<br>"
                f"<strong>Context:</strong> {', '.join([f'Task {idx+1}' for idx in task.get('context_indexes', [])])}<br>"
                f"<strong>Tools:</strong> {list(filter(lambda tool: tool in active_tools, task.get('tools', [])))}</div></div>"
                if 0 <= task['agent_index'] < len(st.session_state.new_agents) else
                f"<div style='margin-bottom: 10px;'><strong>{task['description']}</strong><br>"
                f"<div style='margin-left: 20px;'>"
                f"<strong>Agent:</strong> Invalid Agent Index<br>"
                f"<strong>Expected Output:</strong> {task['expected_output']}<br>"
                f"<strong>Context:</strong> {', '.join([f'Task {idx+1}' for idx in task.get('context_indexes', [])])}<br>"
                f"<strong>Tools:</strong> {list(filter(lambda tool: tool in active_tools, task.get('tools', [])))}</div></div>"
                for task in st.session_state.new_tasks
            )
            st.markdown(f"""
            <style>.crewai-description {{ font-size: 11px; font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; padding-left: 20px; }}
                .crewai-ul {{ font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; list-style-type: disc; padding-left: 20px; }}
                .crewai-li {{ margin-bottom: 10px; font-size: 11px !important; }}
                .crewai-img {{ width: 69px; margin-bottom: 10px; }}</style>
            {task_list_items}
            """, unsafe_allow_html=True)

# Updated create_new_agent_form
@st.experimental_fragment
def create_new_agent_form():
    agent_form = st.empty()
    agent_list_container = st.empty()  # Container for the agent list
    active_tools = st.session_state.get("active_tools", [])
    with agent_form.form(key="agent_form", clear_on_submit=True):
        st.subheader("Add Agent")
        role = st.text_input("Agent Name", key="role_input", placeholder="Research Analyst")
        tools = st.multiselect("Tools", active_tools, key="tools_input")
        goal = st.text_area("Goal", key="goal_input", placeholder="Analyze the company website and provided description to extract insights on culture, values, and specific needs.")
        backstory = st.text_area("Backstory", key="backstory_input", placeholder="Expert in analyzing company cultures and identifying key values and needs from various sources, including websites and brief descriptions.")
        llm = st.text_input("LLM", value=st.session_state.current_llm, key="llm_input")
        allow_delegation = st.toggle("Allow Delegation", value=False, key="allow_delegation_input")
        memory = st.toggle("Memory", value=True, key="memory_input")
        if st.form_submit_button("Save Agent"):
            st.session_state.new_agents.append({
                "role": role,
                "goal": goal,
                "backstory": backstory,
                "llm": llm,
                "allow_delegation": allow_delegation,
                "memory": memory,
                "tools": tools
            })
            agent_form.empty()
            st.session_state.show_agent_form = False
            update_agent_list(agent_list_container)
            st.rerun()
            return True
    return False

# Updated create_new_task_form
@st.experimental_fragment
def create_new_task_form():
    task_form = st.empty()
    task_list_container = st.empty()  # Container for the task list
    active_tools = st.session_state.get("active_tools", [])
    with task_form.form(key="task_form", clear_on_submit=True):
        st.subheader("Add Task")
        description = st.text_area("Description", key="description_input", placeholder=
            """Analyze the provided company website and the hiring manager's company's domain, description. Focus on understanding the company's culture, values, and mission. Identify unique selling points and specific projects or achievements highlighted on the site.Compile a report summarizing these insights, specifically how they can be leveraged in a job posting to attract the right candidates.""")
        tools = st.multiselect("Tools", active_tools, key="task_tools_input")
        if st.session_state.new_agents:
            agent_index = st.number_input("Agent Index for Task", min_value=0, max_value=len(st.session_state.new_agents) - 1, step=1, key="agent_index_input")
        else:
            agent_index = 0
        expected_output = st.text_area("Expected Output", key="expected_output_input", placeholder="""A comprehensive report detailing the company's culture, values, and mission, along with specific selling points relevant to the job role. Suggestions on incorporating these insights into the job posting should be included.""")
        context_indexes = st.multiselect("Context Task Indexes for Task", list(range(len(st.session_state.new_tasks))), key="context_indexes_input")
        if st.form_submit_button("Save Task"):
            st.session_state.new_tasks.append({
                "description": description,
                "agent_index": agent_index,
                "expected_output": expected_output,
                "context_indexes": context_indexes,
                "tools": tools
            })
            task_form.empty()
            st.session_state.show_task_form = False
            update_task_list(task_list_container)
            st.rerun()
            return True
    return False

# Function to update the JSON file with new crew info
def update_crew_json(crew_data):
    file_path = 'crewai_crews/crews.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            crews = json.load(file)
    else:
        crews = []
    
    if not any(crew['name'] == example_crew_data['name'] for crew in crews):
        crews.append(example_crew_data)
    
    crews.append(crew_data)
    
    with open(file_path, 'w') as file:
        json.dump(crews, file, indent=4)

def delete_crew(index):
    # Remove the crew from session state
    st.session_state.crew_list.pop(index)
    
    # Update the JSON file
    file_path = 'crewai_crews/crews.json'
    with open(file_path, 'w') as file:
        json.dump(st.session_state.crew_list, file, indent=4)

    # Update the selected state
    st.session_state.crewai_crew_selected.pop(index)

# Function to create new crew container
def create_new_crew_container():
    with st.container(border=True):
        tab1, tab2, tab3, tab4 = st.tabs(["Tools", "Agents", "Tasks", "Crew"])

        # Tools Tab
        with tab1:
            st.header("Create New Crew")
            st.subheader("Tools")
            tools_data = {
                "Active": [False for _ in TOOLS],
                "Tool": [tool["name"] for tool in TOOLS],
                "Source": [tool["source"] for tool in TOOLS],
                "Description": [tool["description"] for tool in TOOLS],
                "Requires API Key": [tool["needsApiKey"] for tool in TOOLS]
            }
            tools_df = pd.DataFrame(tools_data)

            # Displaying the dataframe with st.data_editor
            edited_tools_df = st.data_editor(
                tools_df,
                height=777,
                column_config={
                    "Active": st.column_config.CheckboxColumn("Active", width=120),
                    "Tool": st.column_config.TextColumn("Tool", disabled=True, width=300),
                    "Source": st.column_config.TextColumn("Source", disabled=True, width=100),
                    "Description": st.column_config.TextColumn("Description", disabled=True, width=1200),
                    "Requires API Key": st.column_config.CheckboxColumn("Req. API Key", disabled=True, width=120)
                },
                num_rows="fixed", # Update to "dynamic" if you want to add a new row (Could be for custom tools) *Not implemented
                key="tools_editor"
            )

            # Update session state with active tools
            current_active_tools = edited_tools_df[edited_tools_df["Active"]]["Tool"].tolist()

            # Check if there is any change in active tools
            if "active_tools" not in st.session_state or st.session_state.active_tools != current_active_tools:
                st.session_state.active_tools = current_active_tools
                st.rerun()

        # Agents Tab
        with tab2:
            st.header("Create New Crew")
            st.subheader("Agents")
            agent_list_container = st.container()
            create_new_agent_form()
            update_agent_list(agent_list_container)

        # Tasks Tab
        with tab3:
            st.header("Create New Crew")
            st.subheader("Tasks")
            task_list_container = st.container()

            if not st.session_state.new_agents:
                st.warning("Please add an agent before creating a task.")
            else:
                create_new_task_form()
                update_task_list(task_list_container)

        # Crew Tab
        with tab4:
            st.header("Create New Crew")
            crew_name = st.text_input("Crew Name", placeholder="Business Requirements Crew")
            agents = st.multiselect("Add Agents", st.session_state.new_agents, format_func=lambda agent: agent['role'], key="crew_agents_multiselect")
            tasks = st.multiselect("Add Tasks", st.session_state.new_tasks, format_func=lambda task: task['description'], key="crew_tasks_multiselect")

            if st.button("Create Crew", key="create_crew_button"):
                if not agents:
                    st.warning("Please add at least one agent before creating a crew.", icon="⚠️")
                elif not tasks:
                    st.warning("Please add at least one task before creating a crew.", icon="⚠️")
                else:
                    crew_data = {
                        "name": crew_name,
                        "agents": agents,
                        "tasks": tasks
                    }
                    st.session_state.crew_list.append(crew_data)
                    update_crew_json(crew_data)
                    reset_form_states()
                    st.rerun()

# Add container for crew list in the sidebar
with st.sidebar:
    with st.container(border=True):
        col1, col2 = st.columns([2, 2])
        st.subheader("***crew***:red[***ai***] Crews")
        st.caption("**Crews will run in sequential order from top to bottom**")

        if st.button("➕ Create a Crew", key="create_new_crew_button", help="Create New Crew"):
            st.session_state.show_crew_container = True

        for i, crew in enumerate(st.session_state.crew_list):
            agent_list_items = "".join(
                f"<div style='margin-bottom: 10px;'><strong>{agent['role']}</strong><br>"
                f"<div style='margin-left: 20px;'><strong>Goal:</strong> {agent['goal']}<br>"
                f"<strong>Backstory:</strong> {agent['backstory']}<br>"
                f"<strong>LLM:</strong> {st.session_state.current_llm}<br>"
                f"<strong>Allow Delegation:</strong> {agent['allow_delegation']}<br>"
                f"<strong>Memory:</strong> {agent['memory']}<br>"
                f"<strong>Tools:</strong> {agent.get('tools', [])}</div></div>"
                for agent in crew["agents"]
            )
            task_list_items = "".join(
                f"<div style='margin-bottom: 10px; margin-left: 20px;'><strong>Description:</strong> {task['description']}<br>"
                f"<div><strong>Agent:</strong> {st.session_state.new_agents[task['agent_index']]['role']}<br>"
                f"<strong>Expected Output:</strong> {task['expected_output']}<br>"
                f"<strong>Context:</strong> {', '.join([f'Task {idx+1}' for idx in task.get('context_indexes', [])])}<br>"
                f"<strong>Tools:</strong> {', '.join(task.get('tools', [])) if 'tools' in task else 'N/A'}</div></div>"
                if 0 <= task['agent_index'] < len(st.session_state.new_agents) else
                f"<div style='margin-bottom: 10px; margin-left: 20px;'><strong>Description:</strong> {task['description']}<br>"
                f"<div><strong>Agent:</strong> Invalid Agent Index<br>"
                f"<strong>Expected Output:</strong> {task['expected_output']}<br>"
                f"<strong>Context:</strong> {', '.join([f'Task {idx+1}' for idx in task.get('context_indexes', [])])}<br>"
                f"<strong>Tools:</strong> {', '.join(task.get('tools', [])) if 'tools' in task else 'N/A'}</div></div>"
                for task in crew["tasks"]
            )
            with st.expander(crew["name"], expanded=False):
                # Update toggle initialization to avoid index error
                if i >= len(st.session_state.crewai_crew_selected):
                    st.session_state.crewai_crew_selected.append(False)
                st.session_state.crewai_crew_selected[i] = st.toggle(f"Run Crew: {crew['name']}", key=f"crew_{i}_selected", value=st.session_state.crewai_crew_selected[i])
                st.html(f"""
                <style>.crewai-description {{ font-size: 11px; font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; padding-left: 20px; }}
                    .crewai-ul {{ font-family: 'Roboto', 'Open Sans', 'Arial', sans-serif; list-style-type: disc; padding-left: 20px; }}
                    .crewai-li {{ margin-bottom: 10px; font-size: 11px !important; }}
                    .crewai-img {{ width: 69px; margin-bottom: 10px; }}</style>
                <p><strong>Agents</strong></p>
                {agent_list_items}
                <p><strong>Tasks</strong></p>
                {task_list_items}
                """)
                if st.button("Delete Crew", key=f"delete_crew_{i}"):
                    delete_crew(i)
                    st.rerun()

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

            response_chunks = []
            async for chunk in chain.astream({"query": user_query, "context": context, "history": chat_messages_history}):
                response_chunks.append(chunk)
                #st.write(chunk, end="", flush=True)

            response = "".join(response_chunks)
            return response
    except Exception as e:
        st.error(f"Error in get_response_async: {e}")
        return ""

def get_response(llm, user_query, tool, chat_messages_history, context=""):
    return asyncio.run(get_response_async(llm, user_query, tool, chat_messages_history, context))

# Display create new crew form
if st.session_state.show_crew_container:
    create_new_crew_container()
else:
    # Handle document upload
    handle_document_upload(langchain_upload_docs_selected)

    # Display chat history and handle user input
    for msg in chat_messages_history.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


    # Handle user chat input
    user_input = st.chat_input("What's on your mind?", key="user_input")
    if user_input:
        st.chat_message("user").write(user_input)
        chat_messages_history.add_user_message(user_input)

        if langchain_upload_docs_selected and st.session_state.vectorstore:
            if st.session_state.qa_chain:
                response = st.session_state.qa_chain.invoke({"input": user_input})
                answer = response.get('answer', "No documents found, please upload documents.")
                context = response.get('context', [])
                st.chat_message("assistant").write(answer)
                chat_messages_history.add_ai_message(answer)
                if context:
                    unique_sources = {f"{doc.metadata['source']}" for doc in context if 'source' in doc.metadata}
                    with st.expander("Sources", False):
                        for source_info in unique_sources:
                            st.write(f"- {source_info}")
            else:
                st.error("QA Chain is not initialized. Please check the configuration.")
        else:
            response = get_response(st.session_state.llm, user_input, "", chat_messages_history, context="")
            chat_messages_history.add_ai_message(response)
            st.chat_message("assistant").write(response)

        for i, selected in enumerate(st.session_state.crewai_crew_selected):
            if selected:
                with st.container():
                    start_time = time.time()
                    with st.chat_message("assistant"):
                        with st.spinner(f"{st.session_state.crew_list[i]['name']} Working..."):
                            with st.expander(f"{st.session_state.crew_list[i]['name']} Interaction Logs", False):

                                # Format output text and color coding for different agents
                                sys.stdout = StreamToExpander(st, agent_task_outputs)

                                dynamic_crew_handler = DynamicCrewHandler(
                                    name=st.session_state.crew_list[i]["name"],
                                    agents=st.session_state.crew_list[i]["agents"],
                                    tasks=st.session_state.crew_list[i]["tasks"],
                                    llm=st.session_state.llm,
                                    user_prompt=user_input,
                                    chat_history=chat_messages_history
                                )
                                response, new_crew_data = dynamic_crew_handler.create_crew()

                                # Format and display the task descriptions
                                task_descriptions = [task['description'] for task in st.session_state.crew_list[i]['tasks']]
                                task_df = pd.DataFrame(task_descriptions, columns=["Tasks"])
                                st.dataframe(task_df, hide_index=True)

                                # Format and display the agent roles
                                agent_roles = [agent['role'] for agent in st.session_state.crew_list[i]['agents']]
                                agent_df = pd.DataFrame(agent_roles, columns=["Agents"])
                                st.dataframe(agent_df, hide_index=True)
                        st.empty().text(f"Total Time Elapsed: {time.time() - start_time:.2f} seconds")

                    with st.chat_message("assistant"):
                        if isinstance(response, tuple):
                            response_text, crew_info = response
                            st.write(response_text)
                            st.json(crew_info)
                        else:
                            response_text = response
                            st.write(response_text)
                        chat_messages_history.add_ai_message(response_text)
                        st.session_state.messages.append({"role": "assistant", "content": response_text})

