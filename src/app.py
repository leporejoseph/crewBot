# src/app.py
import streamlit as st
from streamlit_card import card
from utils.llm_handler import set_initial_llm, update_api_key, toggle_selection, get_response
from utils.document_handler import handle_document_upload
from utils.streamlit_expander import StreamToExpander
from crew_ai.crewai_utils import DynamicCrewHandler, TOOLS, update_crew_json, delete_crew
from config import init_session_state, llm_options, chat_messages_history, agent_colors, get_card_styles, get_empty_card_styles, initialize_app
import os, sys, json, pandas as pd, random, time, re

initialize_app()
init_session_state()
set_initial_llm()

TOOL_NAMES = [tool["name"] for tool in TOOLS]

st.markdown("""<style>body { font-family: Arial, sans-serif; } h2 { color: #CD4055; }</style>""", unsafe_allow_html=True)
crewai_edit_dialog = st.empty()

# Sidebar Configuration
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

# Form and Dialog Handling
def reset_form_states():
    for key in ['show_crew_container', 'show_agent_form', 'show_task_form']:
        st.session_state[key] = False

@st.experimental_fragment
def show_agent_form(): st.session_state.show_agent_form = True

@st.experimental_fragment
def show_task_form(): st.session_state.show_task_form = True

@st.experimental_dialog("Edit Agent")
def edit_agent_dialog(agent, agent_index, crew_index):
    st.session_state.dialog_open = True  
    current_llm = st.session_state.current_llm
    llm_index = llm_options.index(current_llm)

    if st.session_state.dialog_open:
        with st.container(border=False):
            role = st.text_input("Agent Name", value=agent["role"], key=f"agent_role_{agent_index}")
            tools = st.multiselect("Tools", TOOL_NAMES, default=agent.get("tools", []), key=f"agent_tools_{agent_index}")
            goal = st.text_area("Goal", value=agent["goal"], key=f"agent_goal_{agent_index}")
            backstory = st.text_area("Backstory", value=agent["backstory"], key=f"agent_backstory_{agent_index}")
            llm = st.selectbox("LLM", llm_options, index=llm_index, key=f"agent_llm_{agent_index}", disabled=True)
            allow_delegation = st.toggle("Allow Delegation", value=agent.get("allow_delegation", False), key=f"agent_allow_delegation_{agent_index}")
            memory = st.toggle("Memory", value=agent.get("memory", True), key=f"agent_memory_{agent_index}")

            st.session_state.saved_agent_clicked = False
            if st.button(label="Save Agent"):
                st.session_state.saved_agent_clicked = True
                update_agent(agent_index, crew_index, role, goal, backstory, llm, allow_delegation, memory, tools)
                st.session_state.dialog_open = False
                st.rerun()

@st.experimental_dialog("Edit Task")
def edit_task_dialog(task, task_index):
    st.session_state.dialog_open = True
    crew = next((crew for crew in st.session_state.crew_list if task in crew["tasks"]), None)
    if not crew:
        st.error("Task not found in any crew.")
        return
    
    current_agents = crew["agents"]
    agent_names = [agent['role'] for agent in current_agents]

    if st.session_state.dialog_open:
        #with st.form(key=f"edit_task_form_{task_index}", border=False):
        with st.container(border=False):
            st.write(f"Edit Task {task_index + 1}")
            description = st.text_area("Description", value=task["description"], key=f"task_description_{task_index}")
            tools = st.multiselect("Tools", TOOL_NAMES, default=task.get("tools", []), key=f"task_tools_{task_index}")
            assigned_agent = st.selectbox("Assigned Agent", agent_names, index=task["agent_index"], key=f"task_agent_{task_index}")
            expected_output = st.text_area("Expected Output", value=task.get("expected_output", ""), key=f"task_expected_output_{task_index}")
            context_task_options = [f'Task {i+1}' for i in range(len(crew["tasks"]))]
            default_context_tasks = [f'Task {i+1}' for i in task.get("context_indexes", [])]
            context_indexes = st.multiselect("Context Task Indexes for Task", options=context_task_options, default=default_context_tasks, key=f"task_context_indexes_{task_index}")

            st.session_state.saved_task_clicked = False
            if st.form_submit_button(label="Save Task"):
                st.session_state.saved_task_clicked = True
                update_task(task_index, crew, description, assigned_agent, expected_output, context_indexes, tools)
                st.session_state.dialog_open = False
                st.rerun() 
    
if st.session_state.get("saved_task_clicked", False) or st.session_state.get("saved_agent_clicked", False):
    crewai_edit_dialog = st.empty()
    st.session_state.saved_task_clicked = False
    st.session_state.saved_agent_clicked = False
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
                    on_click=lambda idx=agent_index: (handle_card_click(agent, idx, is_agent=False), update_dialog()),
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
                    on_click=lambda tidx=task_index: (handle_card_click(task, tidx, is_agent=False), update_dialog()),
                    styles=get_card_styles(color_index),
                    key=f"task-card-{task_index}-{random.randint(1, 100000)}"
                )
        with task_cols[(len(st.session_state.new_tasks)) % 5]:
            card(
                title="No Tasks Added" if len(st.session_state.new_tasks) == 0 else "",
                text="Click Here to Add an Task" if len(st.session_state.new_agents) == 0 else "Click Here to Add Another Task",
                on_click=show_task_form,
                styles=get_empty_card_styles(),
                key=f"task-card-add-new-{len(st.session_state.new_tasks)}"
            )
        if st.session_state.show_task_form:
            create_new_task_form(container)

@st.experimental_fragment
def handle_card_click(agent_or_task, agent_or_task_index, crew_index, is_agent=True):
    if is_agent:
        if not st.session_state.get("saved_agent_clicked", False):
            edit_agent_dialog(agent_or_task, agent_or_task_index, crew_index)
    else:
        if not st.session_state.get("saved_task_clicked", False):
            edit_task_dialog(agent_or_task, agent_or_task_index, crew_index)

@st.experimental_fragment
def update_agent(agent_index, crew_index, role, goal, backstory, llm, allow_delegation, memory, tools):
    st.session_state.crew_list[crew_index]["agents"][agent_index] = {
        "role": role, 
        "goal": goal, 
        "backstory": backstory, 
        "llm": llm,
        "allow_delegation": allow_delegation, 
        "memory": memory, 
        "tools": tools
    }
    # Update the crews.json file
    update_crew_json(st.session_state.crew_list)

@st.experimental_fragment
def update_task(task_index, crew, description, assigned_agent, expected_output, context_indexes, tools):
    crew["tasks"][task_index] = {
        "description": description, "agent_index": assigned_agent,
        "expected_output": expected_output,
        "context_indexes": [int(idx.split()[-1]) - 1 for idx in context_indexes],
        "tools": tools
    }

def create_new_agent_form(agent_list_container):
    agent_form = st.empty()
    active_tools = st.session_state.get("active_tools", [])
    with agent_form.form(key="agent_form", clear_on_submit=True, border=False):
    #with st.container(border=False):
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

def create_new_task_form(task_list_container):
    active_tools = st.session_state.get("active_tools", [])
    agent_names = [agent['role'] for agent in st.session_state.new_agents]

    with st.form(key="task_form", clear_on_submit=True):
    #with st.container(border=False):
        st.subheader("Add Task")
        description = st.text_area("Description", key="description_input", placeholder="Analyze the provided company website and the hiring manager's company's domain, description. Focus on understanding the company's culture, values, and mission. Identify unique selling points and specific projects or achievements highlighted on the site.Compile a report summarizing these insights, specifically how they can be leveraged in a job posting to attract the right candidates.")
        tools = st.multiselect("Tools", active_tools, key="task_tools_input")
        assigned_agent = st.selectbox("Assigned Agent", agent_names, key="agent_name_input")
        expected_output = st.text_area("Expected Output", key="expected_output_input", placeholder="A comprehensive report detailing the company's culture, values, and mission, along with specific selling points relevant to the job role. Suggestions on incorporating these insights into the job posting should be included.")
        context_indexes = st.multiselect("Context for Task", [f'Task {i+1}' for i in range(len(st.session_state.new_tasks))], key="context_indexes_input")

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

@st.experimental_fragment
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
            with st.form(key="crew_form", clear_on_submit=True, border=False):
                crew_name = st.text_input("Crew Name", placeholder="Business Requirements Crew")
                agents = st.multiselect("Add Agents", st.session_state.new_agents, format_func=lambda agent: agent['role'], key="crew_agents_multiselect")
                tasks = st.multiselect("Add Tasks", st.session_state.new_tasks, format_func=lambda task: task['description'], key="crew_tasks_multiselect")

                if st.form_submit_button("Create Crew"):
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

def update_dialog():
    global crewai_edit_dialog
    crewai_edit_dialog = st.empty()

# Main Execution
if st.session_state.show_crew_container:
    create_new_crew_container()
else:
    handle_document_upload(st.session_state.langchain_upload_docs_selected)

    for msg in chat_messages_history.messages:
        if msg.type == "ai" and "[DEBUG]:" in msg.content:
            crew_name_match = re.findall(r"\[crewai_expander\]:==\s*(.*?)(?:\s|$)", msg.content, re.IGNORECASE)
            crew_start_time = re.findall(r"\[crewai_starttime\]:==\s*(.*?)(?:\s|$)", msg.content, re.IGNORECASE)
            crew_end_time = re.findall(r"\[crewai_endtime\]:==\s*(.*?)(?:\s|$)", msg.content, re.IGNORECASE)
            if crew_name_match:
                crew_name = crew_name_match[0]
                with st.chat_message("assistant"):
                    with st.expander(f"{crew_name} Interaction Logs", expanded=False):
                        st.write(msg.content)
                        # Create dataframes for agent and task details
                        crew_info = next((crew for crew in st.session_state.crew_list if crew['name'] == crew_name), None)
                        if crew_info:
                            agent_df = pd.DataFrame([agent['role'] for agent in crew_info['agents']], columns=["Agents"])
                            st.dataframe(agent_df, hide_index=True)
                            task_df = pd.DataFrame([task['description'] for task in crew_info['tasks']], columns=["Tasks"])
                            st.dataframe(task_df, hide_index=True)
                    # Convert start and end times from strings to floats
                    if crew_start_time and crew_end_time:
                        start_time = float(crew_start_time[0])
                        end_time = float(crew_end_time[0])
                        elapsed_time = end_time - start_time
                        st.empty().text(f"Total Time Elapsed: {elapsed_time:.2f} seconds")
        else:
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

        all_crews_finished = False
        for i, selected in enumerate(st.session_state.crewai_crew_selected):
            if selected:
                start_time = time.time()
                st.session_state[f"{st.session_state.crew_list[i]['name']}_start_time"] = start_time
                crew_name = st.session_state.crew_list[i]['name']
                agents = st.session_state.crew_list[i]['agents']
                with st.chat_message("assistant"):
                    with st.expander(f"{crew_name} Interaction Logs", False):
                        with st.spinner(f"{crew_name} Working..."):
                            sys.stdout = StreamToExpander(st, crew_name, agents)

                            dynamic_crew_handler = DynamicCrewHandler(
                                name=crew_name,
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
                        elapsed_time = time.time() - start_time
                        st.empty().text(f"Total Time Elapsed Test: {elapsed_time:.2f} seconds")
                        all_crews_finished = True
                with st.chat_message("assistant"):
                    response_text = response[0] if isinstance(response, tuple) else response
                    st.write(response_text)
                    chat_messages_history.add_ai_message(response_text)

        if all_crews_finished:
            for i in range(len(st.session_state.crewai_crew_selected)):
                if st.session_state.crewai_crew_selected[i]:
                    toggle_key = f"crew_{i}_selected"
                    st.session_state[toggle_key] = False

@st.experimental_fragment
def display_crew_list():
    with st.sidebar:
        with st.container(border=True):
            st.image('crew_ai/crewai.png', width=150)
            st.caption("**Crews will run in sequential order from top to bottom**")

            if st.button("➕ Create a Crew", key="create_new_crew_button"):
                st.session_state.show_crew_container = True
                st.rerun()

            for i, crew in enumerate(st.session_state.crew_list):
                with st.expander(crew["name"], expanded=False):
                    if i >= len(st.session_state.crewai_crew_selected):
                        st.session_state.crewai_crew_selected.append(False)
                    st.session_state.crewai_crew_selected[i] = st.toggle(f"Run Crew: {crew['name']}", key=f"crew_{i}_selected", value=False)

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
                                on_click=lambda idx=agent_index, cidx=i: (handle_card_click(agent, idx, cidx, is_agent=True), update_dialog()),
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
                                on_click=lambda tidx=task_index, cidx=i: (handle_card_click(task, tidx, cidx, is_agent=False), update_dialog()),
                                styles=get_card_styles(color_index),
                                key=f"task-card-{i}-{task_index}"
                            )

                    if st.button("Delete Crew", key=f"delete_crew_{i}"):
                        delete_crew(i)
                        st.rerun()

display_crew_list()