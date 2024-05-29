# src/app.py
import streamlit as st
from streamlit_card import card
from dotenv import load_dotenv
from utils.llm_handler import set_initial_llm, update_api_key, toggle_selection
from utils.document_handler import handle_document_upload
from utils.streamlit_expander import StreamToExpander
from crew_ai.crewai_utils import DynamicCrewHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from config import init_session_state
import os, sys, json, pandas as pd, random, time, asyncio

def initialize_app():
    os.makedirs('files', exist_ok=True)
    os.makedirs('chromadb', exist_ok=True)
    st.set_page_config(page_title="CrewBot: Your AI Assistant", page_icon="🤖", layout="wide")
    st.title("CrewBot2: Your AI Assistant")
    st.sidebar.title("Configuration")
    load_dotenv()

def initialize_session_state():
    session_defaults = {
        'crew_list': json.load(open('crew_ai/crews.json')),
        'crewai_crew_selected': [False] * len(st.session_state.get('crew_list', [])),
        'tools': [], 'new_agents': [], 'show_agent_form': False, 'show_crew_container': False,
        'show_task_form': False, 'new_tasks': [], 'show_apikey_toggle': False, 'dialog_open': False,
        'langchain_upload_docs_selected': False, 'langchain_export_pdf_selected': False
    }
    for key, value in session_defaults.items():
        st.session_state.setdefault(key, value)

initialize_app()
init_session_state()
set_initial_llm()
initialize_session_state()

llm_options = ["OpenAI", "LM Studio"]
chat_messages_history = StreamlitChatMessageHistory(key='chat_messages')
agent_colors = ["#32CD32", "#20B2AA", "#FFA500", "#FF6347", "#800080", "#1E90FF"]
TOOLS = [{"name": tool, "needsApiKey": False, "source": "crewai", "description": f"A RAG tool for {tool.lower().replace('searchtool', 'searching within ')}"} for tool in [
    "CSVSearchTool", "CodeDocsSearchTool", "DOCXSearchTool", "DirectoryReadTool", "DirectorySearchTool", "FileReadTool", "GithubSearchTool", "JSONSearchTool",
    "MDXSearchTool", "PDFSearchTool", "PGSearchTool", "RagTool", "ScrapeElementFromWebsiteTool", "ScrapeWebsiteTool", "SerperDevTool", "TXTSearchTool",
    "WebsiteSearchTool", "XMLSearchTool", "YoutubeChannelSearchTool", "YoutubeVideoSearchTool"]]
TOOL_NAMES = [tool["name"] for tool in TOOLS]

st.markdown("""<style>body { font-family: Arial, sans-serif; } h2 { color: #CD4055; }</style>""", unsafe_allow_html=True)

def sidebar_configuration():
    with st.sidebar.expander("**LLM Selection**", True):
        llm_selected = st.selectbox("LLM", llm_options, index=0)
        st.session_state.openai_llm_selected = (llm_selected == "OpenAI")
        st.session_state.lmStudio_llm_selected = (llm_selected == "LM Studio")
        toggle_selection("openai_llm_selected" if st.session_state.openai_llm_selected else "lmStudio_llm_selected")

        if st.session_state.openai_llm_selected:
            st.session_state.current_llm = "OpenAI"
            st.text_input('Model', value=st.session_state.get("openai_api_model", "gpt-3.5-turbo"), key='openai_api_model', on_change=set_initial_llm)
            st.session_state.show_apikey_toggle = st.toggle("Show Api Key", value=False, key='show_openai_key')
            if st.session_state.show_apikey_toggle:
                st.text_input('OpenAI API Key', os.getenv("OPENAI_API_KEY"), on_change=update_api_key, key='openai_api_key')
        else:
            st.session_state.current_llm = "LM Studio"
            st.text_input('Model', value=st.session_state.get("lm_studio_model", ""), key='lm_studio_model', on_change=set_initial_llm)
            st.text_input('Base URL', value=st.session_state.get("lm_studio_base_url", ""), key='lm_studio_base_url', on_change=set_initial_llm)

    with st.sidebar.expander("LangChain Tools :parrot: :link:", True):
        st.session_state.langchain_upload_docs_selected = st.toggle("Upload Documents", value=True)
        st.session_state.langchain_export_pdf_selected = st.toggle("Custom Tool: Export PDF", value=False)

sidebar_configuration()

def reset_form_states():
    for key in ['show_crew_container', 'show_agent_form', 'show_task_form']:
        st.session_state[key] = False

def close_all_dialogs():
    for key in ['show_agent_form', 'show_task_form', 'show_crew_container']:
        st.session_state[key] = False

@st.experimental_fragment
def show_agent_form(): st.session_state.show_agent_form = True

@st.experimental_fragment
def show_task_form(): st.session_state.show_task_form = True

@st.experimental_dialog("Edit Agent")
def edit_agent_dialog(agent, agent_index):
    st.session_state.dialog_open = True  
    close_all_dialogs()
    current_llm = st.session_state.current_llm
    llm_index = llm_options.index(current_llm)

    with st.form(key=f"edit_agent_form_{agent_index}", border=False):
        role = st.text_input("Agent Name", value=agent["role"], key=f"agent_role_{agent_index}")
        tools = st.multiselect("Tools", TOOL_NAMES, default=agent.get("tools", []), key=f"agent_tools_{agent_index}")
        goal = st.text_area("Goal", value=agent["goal"], key=f"agent_goal_{agent_index}")
        backstory = st.text_area("Backstory", value=agent["backstory"], key=f"agent_backstory_{agent_index}")
        llm = st.selectbox("LLM", llm_options, index=llm_index, key=f"agent_llm_{agent_index}", disabled=True)
        allow_delegation = st.toggle("Allow Delegation", value=agent.get("allow_delegation", False), key=f"agent_allow_delegation_{agent_index}")
        memory = st.toggle("Memory", value=agent.get("memory", True), key=f"agent_memory_{agent_index}")

        if st.form_submit_button(label="Save Agent"):
            update_agent(agent_index, role, goal, backstory, llm, allow_delegation, memory, tools)
            st.session_state.dialog_open = False 
            st.rerun()

@st.experimental_dialog("Edit Task")
def edit_task_dialog(task, task_index):
    st.session_state.dialog_open = True
    close_all_dialogs()
    crew = next((crew for crew in st.session_state.crew_list if task in crew["tasks"]), None)
    if not crew:
        st.error("Task not found in any crew.")
        return
    
    current_agents = crew["agents"]
    agent_names = [agent['role'] for agent in current_agents]

    with st.form(key=f"edit_task_form_{task_index}", border=False):
        st.write(f"Edit Task {task_index + 1}")
        description = st.text_area("Description", value=task["description"], key=f"task_description_{task_index}")
        tools = st.multiselect("Tools", TOOL_NAMES, default=task.get("tools", []), key=f"task_tools_{task_index}")
        assigned_agent = st.selectbox("Assigned Agent", agent_names, index=task["agent_index"], key=f"task_agent_{task_index}")
        expected_output = st.text_area("Expected Output", value=task.get("expected_output", ""), key=f"task_expected_output_{task_index}")
        context_task_options = [f'Task {i+1}' for i in range(len(crew["tasks"]))]
        default_context_tasks = [f'Task {i+1}' for i in task.get("context_indexes", [])]
        context_indexes = st.multiselect("Context Task Indexes for Task", options=context_task_options, default=default_context_tasks, key=f"task_context_indexes_{task_index}")

        if st.form_submit_button(label="Save Task"):
            update_task(task_index, crew, description, assigned_agent, expected_output, context_indexes, tools)
            st.session_state.dialog_open = False 
            st.rerun()

def update_agent_list(container):
    with container:
        agent_cols = st.columns(5)
        for agent_index, agent in enumerate(st.session_state.new_agents):
            color_index = agent_index % len(agent_colors)
            with agent_cols[agent_index % 5]:
                card(
                    title=agent['role'],
                    text=f"Tools: {len(agent.get('tools', [])) or 'N/A'}",
                    on_click=lambda idx=agent_index: handle_card_click(agent, idx, is_agent=True),
                    styles=get_card_styles(color_index),
                    key=f"agent-card-{agent_index}-{random.randint(1, 100000)}"
                )
        with agent_cols[(len(st.session_state.new_agents)) % 5]:
            card(
                title="No Agents Added" if len(st.session_state.new_agents) == 0 else "",
                text="Click Here to Add an Agent" if len(st.session_state.new_agents) == 0 else "Click Here to Add Another Agent",
                on_click=show_agent_form,
                styles=get_empty_card_styles(),
                key=f"agent-card-add-new-{len(st.session_state.new_agents)}"
            )
        if st.session_state.show_agent_form:
            create_new_agent_form(container)

def update_task_list(container):
    with container:
        task_cols = st.columns(5)
        for task_index, task in enumerate(st.session_state.new_tasks):
            color_index = task_index % len(agent_colors)
            with task_cols[task_index % 5]:
                card(
                    title=task.get('description', 'No description'),
                    text=f"Tools: {', '.join(task.get('tools', [])) or 'N/A'}\nAgent: {task.get('agent_role', 'No agent role')}\nContext: {', '.join([f'Task {idx+1}' for idx in task.get('context_indexes', [])]) or 'N/A'}",
                    on_click=lambda tidx=task_index: handle_card_click(task, tidx, is_agent=False),
                    styles=get_card_styles(color_index),
                    key=f"task-card-{task_index}-{random.randint(1, 100000)}"
                )
        with task_cols[(len(st.session_state.new_tasks)) % 5]:
            card(
                title="No Tasks added",
                text="Click here to add a task",
                on_click=show_task_form,
                styles=get_empty_card_styles(),
                key=f"task-card-add-new-{len(st.session_state.new_tasks)}"
            )
        if st.session_state.show_task_form:
            create_new_task_form(container)

def handle_card_click(agent_or_task, agent_or_task_index, is_agent=True):
    close_all_dialogs()
    if is_agent:
        edit_agent_dialog(agent_or_task, agent_or_task_index)
    else:
        edit_task_dialog(agent_or_task, agent_or_task_index)

def get_card_styles(color_index):
    return {
        "card": {
            "width": "100%",
            "height": "250px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.1)",
            "margin": "0px",
            "background-color": agent_colors[color_index],
        },
        "title": {
            "font-size": "26px",
        },
        "text": {
            "font-family": "Roboto, Open Sans, Arial, sans-serif",
            "font-size": "18px",
            "padding": "10px",
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
            "height": "250px",
            "border-radius": "10px",
            "box-shadow": "0 0 10px rgba(0,0,0,0.1)",
            "margin": "0px",
            "background-color": "#262730",
        },
        "title": {
            "font-size": "26px",
        },
        "text": {
            "font-family": "Roboto, Open Sans, Arial, sans-serif",
            "font-size": "18px",
            "padding": "10px",
            "overflow": "hidden",
            "text-overflow": "ellipsis",
            "white-space": "nowrap",
            "max-height": "3em"
        }
    }

def update_agent(agent_index, role, goal, backstory, llm, allow_delegation, memory, tools):
    for crew in st.session_state.crew_list:
        if any(agent in crew["agents"] for agent in crew["agents"]):
            crew["agents"][agent_index] = {
                "role": role, "goal": goal, "backstory": backstory, "llm": llm,
                "allow_delegation": allow_delegation, "memory": memory, "tools": tools
            }
            break

def update_task(task_index, crew, description, assigned_agent, expected_output, context_indexes, tools):
    crew["tasks"][task_index] = {
        "description": description, "agent_index": assigned_agent,
        "expected_output": expected_output,
        "context_indexes": [int(idx.split()[-1]) - 1 for idx in context_indexes],
        "tools": tools
    }

@st.experimental_fragment
def create_new_agent_form(agent_list_container):
    agent_form = st.empty()
    active_tools = st.session_state.get("active_tools", [])
    with agent_form.form(key="agent_form", clear_on_submit=True, border=False):
        st.subheader("Add Agent")
        role = st.text_input("Agent Name", key="role_input", placeholder="Research Analyst")
        tools = st.multiselect("Tools", active_tools, key="tools_input")
        goal = st.text_area("Goal", key="goal_input", placeholder="Analyze the company website and provided description to extract insights on culture, values, and specific needs.")
        backstory = st.text_area("Backstory", key="backstory_input", placeholder="Expert in analyzing company cultures and identifying key values and needs from various sources, including websites and brief descriptions.")
        llm = st.text_input("LLM", value=st.session_state.current_llm, key="llm_input", disabled=True)
        allow_delegation = st.toggle("Allow Delegation", value=False, key="allow_delegation_input")
        memory = st.toggle("Memory", value=True, key="memory_input")

        if st.form_submit_button("Add Agent"):
            st.session_state.new_agents.append({
                "role": role, "goal": goal, "backstory": backstory, "llm": llm,
                "allow_delegation": allow_delegation, "memory": memory, "tools": tools
            })
            st.session_state.update({"show_agent_form": False})
            update_agent_list(agent_list_container)
            st.rerun()
    return False

@st.experimental_fragment
def create_new_task_form(task_list_container):
    active_tools = st.session_state.get("active_tools", [])
    agent_names = [agent['role'] for agent in st.session_state.new_agents]

    with st.form(key="task_form", clear_on_submit=True):
        st.subheader("Add Task")
        description = st.text_area("Description", key="description_input", placeholder="Analyze the provided company website and the hiring manager's company's domain, description. Focus on understanding the company's culture, values, and mission. Identify unique selling points and specific projects or achievements highlighted on the site.Compile a report summarizing these insights, specifically how they can be leveraged in a job posting to attract the right candidates.")
        tools = st.multiselect("Tools", active_tools, key="task_tools_input")
        assigned_agent = st.selectbox("Assigned Agent", agent_names, key="agent_name_input")
        expected_output = st.text_area("Expected Output", key="expected_output_input", placeholder="A comprehensive report detailing the company's culture, values, and mission, along with specific selling points relevant to the job role. Suggestions on incorporating these insights into the job posting should be included.")
        context_indexes = st.multiselect("Context Task Indexes for Task", [f'Task {i+1}' for i in range(len(st.session_state.new_tasks))], key="context_indexes_input")

        if st.form_submit_button("Add Task"):
            st.session_state.new_tasks.append({
                "description": description, "agent_index": agent_names.index(assigned_agent),
                "expected_output": expected_output,
                "context_indexes": [int(idx.split()[-1]) - 1 for idx in context_indexes],
                "tools": tools
            })
            st.session_state.update({"show_task_form": False})
            update_task_list(task_list_container)
            st.rerun()
    return False

def update_crew_json(crew_data):
    file_path = 'crew_ai/crews.json'
    crews = json.load(open(file_path)) if os.path.exists(file_path) else []
    crews.append(crew_data)
    json.dump(crews, open(file_path, 'w'), indent=4)

def delete_crew(index):
    st.session_state.crew_list.pop(index)
    json.dump(st.session_state.crew_list, open('crew_ai/crews.json', 'w'), indent=4)
    st.session_state.crewai_crew_selected.pop(index)

def create_new_crew_container():
    with st.container(border=True):
        st.header("Create New Crew")
        tab1, tab2, tab3, tab4 = st.tabs(["Tools", "Agents", "Tasks", "Crew"])

        with tab1:
            st.subheader("Tools")
            tools_data = {
                "Active": [False for _ in TOOLS],
                "Tool": [tool["name"] for tool in TOOLS],
                "Source": [tool["source"] for tool in TOOLS],
                "Description": [tool["description"] for tool in TOOLS],
                "Req. API Key": [tool["needsApiKey"] for tool in TOOLS]
            }
            tools_df = pd.DataFrame(tools_data)
            edited_tools_df = st.data_editor(tools_df, height=777, hide_index=True, column_config={
                "Active": st.column_config.CheckboxColumn("Active", width=100),
                "Tool": st.column_config.TextColumn("Tool", disabled=True, width=300),
                "Source": st.column_config.TextColumn("Source", disabled=True, width=100),
                "Description": st.column_config.TextColumn("Description", disabled=True, width=1000),
                "Req. API Key": st.column_config.CheckboxColumn("Req. API Key", disabled=True, width=100)
            }, num_rows="fixed", key="tools_editor")

            st.session_state.active_tools = edited_tools_df[edited_tools_df["Active"]]["Tool"].tolist()

        with tab2:
            st.subheader("Agents")
            agent_list_container = st.container()
            update_agent_list(agent_list_container)

        with tab3:
            st.subheader("Tasks")
            task_list_container = st.container()
            if not st.session_state.new_agents:
                st.warning("Please add an agent before creating a task.")
            else:
                update_task_list(task_list_container)

        with tab4:
            crew_name = st.text_input("Crew Name", placeholder="Business Requirements Crew")
            agents = st.multiselect("Add Agents", st.session_state.new_agents, format_func=lambda agent: agent['role'], key="crew_agents_multiselect")
            tasks = st.multiselect("Add Tasks", st.session_state.new_tasks, format_func=lambda task: task['description'], key="crew_tasks_multiselect")

            if st.button("Create Crew", key="create_crew_button"):
                if not agents:
                    st.warning("Please add at least one agent before creating a crew.", icon="⚠️")
                elif not tasks:
                    st.warning("Please add at least one task before creating a crew.", icon="⚠️")
                else:
                    crew_data = {"name": crew_name, "agents": agents, "tasks": tasks}
                    st.session_state.crew_list.append(crew_data)
                    update_crew_json(crew_data)
                    reset_form_states()
                    st.rerun()

def display_crew_list():
    with st.sidebar:
        with st.container(border=True):
            st.image('crew_ai/crewai.png', width=150)
            st.caption("**Crews will run in sequential order from top to bottom**")

            if st.button("➕ Create a Crew", key="create_new_crew_button"):
                st.session_state.show_crew_container = True

            for i, crew in enumerate(st.session_state.crew_list):
                with st.expander(crew["name"], expanded=False):
                    if i >= len(st.session_state.crewai_crew_selected):
                        st.session_state.crewai_crew_selected.append(False)
                    st.session_state.crewai_crew_selected[i] = st.toggle(f"Run Crew: {crew['name']}", key=f"crew_{i}_selected", value=st.session_state.crewai_crew_selected[i])

                    agent_col1, task_col2 = st.columns(2)

                    with agent_col1:
                        st.markdown("<h3>Agents</h3>", unsafe_allow_html=True)
                    with task_col2:
                        st.markdown("<h3>Tasks</h3>", unsafe_allow_html=True)

                    for agent_index, agent in enumerate(crew["agents"]):
                        color_index = agent_index % len(agent_colors)
                        with agent_col1:
                            card(
                                title=agent['role'],
                                text=f"Tools: {len(agent.get('tools', [])) or 'N/A'}",
                                on_click=lambda idx=agent_index, cidx=i: edit_agent_dialog(agent, idx, cidx),
                                styles=get_card_styles(color_index),
                                key=f"agent-card-{i}-{agent_index}"
                            )

                    tasks_by_agent = sorted(crew["tasks"], key=lambda x: x['agent_index'])
                    for task_index, task in enumerate(tasks_by_agent):
                        task_agent_role = crew['agents'][task['agent_index']]['role'] if 0 <= task['agent_index'] < len(crew['agents']) else 'Invalid Agent Index'
                        color_index = task['agent_index'] % len(agent_colors)
                        with task_col2:
                            card(
                                title=task['description'],
                                text=f"Agent: {task_agent_role}",
                                on_click=lambda tidx=task_index, cidx=i: edit_task_dialog(task, tidx, cidx),
                                styles=get_card_styles(color_index),
                                key=f"task-card-{i}-{task_index}"
                            )

                    if st.button("Delete Crew", key=f"delete_crew_{i}"):
                        delete_crew(i)
                        st.rerun()

display_crew_list()

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

if st.session_state.show_crew_container:
    create_new_crew_container()
else:
    handle_document_upload(st.session_state.langchain_upload_docs_selected)
    for msg in chat_messages_history.messages:
        st.chat_message(msg.type).write(msg.content)
    user_input = st.chat_input("What's on your mind?", key="user_input")
    if user_input:
        st.chat_message("user").write(user_input)
        chat_messages_history.add_user_message(user_input)
        if st.session_state.langchain_upload_docs_selected and st.session_state.vectorstore:
            if st.session_state.qa_chain:
                response = st.session_state.qa_chain.invoke({"input": user_input})
                st.chat_message("assistant").write(response.get('answer', "No documents found, please upload documents."))
                chat_messages_history.add_ai_message(response.get('answer', "No documents found, please upload documents."))
                with st.expander("Sources", False):
                    for source_info in {f"{doc.metadata['source']}" for doc in response.get('context', []) if 'source' in doc.metadata}:
                        st.write(f"- {source_info}")
            else:
                st.error("QA Chain is not initialized. Please check the configuration.")
        else:
            response = get_response(st.session_state.llm, user_input, "", chat_messages_history, context="")
            chat_messages_history.add_ai_message(response)
            st.chat_message("assistant").write(response)
        for i, selected in enumerate(st.session_state.crewai_crew_selected):
            if selected:
                start_time = time.time()
                crew_logs = st.empty()
                with crew_logs.container():
                    with st.chat_message("assistant"):
                        with st.expander(f"{st.session_state.crew_list[i]['name']} Interaction Logs", False):
                            with st.spinner(f"{st.session_state.crew_list[i]['name']} Working..."):
                                sys.stdout = StreamToExpander(st, [])
                                dynamic_crew_handler = DynamicCrewHandler(
                                    name=st.session_state.crew_list[i]["name"],
                                    agents=st.session_state.crew_list[i]["agents"],
                                    tasks=st.session_state.crew_list[i]["tasks"],
                                    llm=st.session_state.llm,
                                    user_prompt=user_input,
                                    chat_history=chat_messages_history
                                )
                                response, new_crew_data = dynamic_crew_handler.create_crew()
                                agent_df = pd.DataFrame([agent['role'] for agent in st.session_state.crew_list[i]['agents']], columns=["Agents"])
                                st.dataframe(agent_df, hide_index=True)
                                task_df = pd.DataFrame([task['description'] for task in st.session_state.crew_list[i]['tasks']], columns=["Tasks"])
                                st.dataframe(task_df, hide_index=True)
                        st.empty().text(f"Total Time Elapsed: {time.time() - start_time:.2f} seconds")
                    with st.chat_message("assistant"):
                        st.write(response[0] if isinstance(response, tuple) else response)
                        chat_messages_history.add_ai_message(response[0] if isinstance(response, tuple) else response)
