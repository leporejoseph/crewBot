# src/app.py
import os, pandas as pd, random, re, time, sys, json
import streamlit as st
from streamlit_card import card
from utils.llm_handler import set_initial_llm, update_api_key, toggle_selection, get_response
from utils.document_handler import handle_document_upload
from utils.streamlit_expander import StreamToExpander
from crew_ai.crewai_utils import DynamicCrewHandler, TOOLS, update_crew_json, delete_crew
from config import (
    init_session_state,
    llm_options,
    chat_messages_history,
    agent_colors,
    get_card_styles,
    get_empty_card_styles,
    initialize_app,
    save_preferences_on_change,
    save_user_preferences,
    save_chat_history,
    clear_chat_history,
    OPENAI_MODEL,
    LM_STUDIO_MODEL,
    LM_STUDIO_BASE_URL,
    GROQ_MODEL,
)

initialize_app()
init_session_state()
set_initial_llm()

TOOL_NAMES = [tool["name"] for tool in TOOLS]

st.markdown("""<style>body { font-family: Arial, sans-serif; } h2 { color: #CD4055; }</style>""", unsafe_allow_html=True)

#region Sidebar Configuration
def sidebar_configuration():
    with st.sidebar.expander("**LLM Selection**", True):
        llm_selected = st.selectbox("LLM", llm_options, index=llm_options.index(st.session_state.get("current_llm", "OpenAI")))
        st.session_state.current_llm = llm_selected
        save_preferences_on_change()  # This is correct
        selection_states = {"OpenAI": "openai_llm_selected", "LM Studio": "lmStudio_llm_selected", "Groq": "groq_llm_selected"}
        for key, value in selection_states.items():
            st.session_state[value] = (llm_selected == key)
        toggle_selection(selection_states[llm_selected])

        if st.session_state.openai_llm_selected:
            st.text_input('Model', value=st.session_state.get("openai_api_model", OPENAI_MODEL), key='openai_api_model', on_change=save_preferences_on_change)
            st.session_state.show_apikey_toggle = st.toggle("Show API Key", value=st.session_state.get("show_apikey_toggle", False), key='show_openai_key', on_change=save_preferences_on_change)
            if st.session_state.show_apikey_toggle:
                st.text_input('OpenAI API Key', os.getenv("OPENAI_API_KEY", ""), on_change=update_api_key, key='openai_api_key')
        elif st.session_state.lmStudio_llm_selected:
            st.text_input('Model', value=st.session_state.get("lm_studio_model", LM_STUDIO_MODEL), key='lm_studio_model', on_change=save_preferences_on_change)
            st.text_input('Base URL', value=st.session_state.get("lm_studio_base_url", LM_STUDIO_BASE_URL), key='lm_studio_base_url', on_change=save_preferences_on_change)
        elif st.session_state.groq_llm_selected:
            st.text_input('Model Name', value=st.session_state.get("groq_model_name", GROQ_MODEL), key='groq_model_name', on_change=save_preferences_on_change)
            st.session_state.show_apikey_toggle = st.toggle("Show API Key", value=st.session_state.get("show_apikey_toggle", False), key='show_groq_key', on_change=save_preferences_on_change)
            if st.session_state.show_apikey_toggle:
                st.text_input('Groq API Key', os.getenv("GROQ_API_KEY", ""), on_change=update_api_key, key='groq_api_key')

        if st.button("❌ Clear Chat History"):
            clear_chat_history()
            st.toast(":robot_face: Chat History Cleared")

    with st.sidebar.expander("LangChain Tools :parrot: :link:", True):
        st.session_state.langchain_upload_docs_selected = st.toggle("Upload Documents", value=st.session_state.get("langchain_upload_docs_selected", False), on_change=save_preferences_on_change)
        st.session_state.langchain_export_pdf_selected = st.toggle("Summarize and Export PDF", key="export_pdf_selected", value=st.session_state.get("langchain_export_pdf_selected", False), on_change=save_preferences_on_change, help="Summarizes the full chat history and exports it as a PDF file.")

sidebar_configuration()

#endregion

#region CrewAi Create Forms
@st.experimental_fragment
def show_agent_form(): 
    st.session_state.show_agent_form = True

@st.experimental_fragment
def show_task_form(): st.session_state.show_task_form = True

@st.experimental_fragment
def create_new_agent_form():
    active_tools = st.session_state.get("active_tools", [])
    with st.form(key="agent_form", clear_on_submit=True, border=False):
        st.subheader("Add Agent")
        role = st.text_input("Agent Name", key="role_input", placeholder="Research Analyst")
        tools = st.multiselect("Tools", active_tools, key="tools_input")
        goal = st.text_area("Goal", key="goal_input", placeholder="Analyze the company website and provided description to extract insights on culture, values, and specific needs.")
        backstory = st.text_area("Backstory", key="backstory_input", placeholder="Expert in analyzing company cultures and identifying key values and needs from various sources, including websites and brief descriptions.")
        llm = st.text_input("LLM", value=st.session_state.current_llm, key="llm_input", disabled=True)
        allow_delegation = st.toggle("Allow Delegation", value=False, key="allow_delegation_input")

        if st.form_submit_button("Add Agent"):
            st.session_state.new_agents.append({
                "role": role, "goal": goal, "backstory": backstory, "llm": llm,
                "allow_delegation": allow_delegation, "tools": tools
            })
            st.session_state.update({"show_agent_form": False})
            update_agent_list()
            st.rerun()
    return False

@st.experimental_fragment
def create_new_task_form():
    active_tools = st.session_state.get("active_tools", [])
    all_agents = get_all_agents_from_crews() + st.session_state.new_agents
    agent_names = [agent['role'] for agent in all_agents]

    with st.form(key="task_form", clear_on_submit=True, border=False):
        st.subheader("Add Task")
        description = st.text_area("Description", key="description_input", placeholder="Analyze the provided company website and the hiring manager's company's domain, description. Focus on understanding the company's culture, values, and mission. Identify unique selling points and specific projects or achievements highlighted on the site. Compile a report summarizing these insights, specifically how they can be leveraged in a job posting to attract the right candidates.")
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
            update_task_list()
            st.rerun()
    return False

def get_all_agents_from_crews():
    all_agents = []
    for crew in st.session_state.crew_list:
        all_agents.extend(crew["agents"])
    return all_agents

def get_all_tasks_from_crews():
    all_tasks = []
    for crew in st.session_state.crew_list:
        all_tasks.extend(crew["tasks"])
    return all_tasks

def truncate_text(text, word_limit):
    words = text.split()
    if len(words) > word_limit:
        return ' '.join(words[:word_limit]) + '...'
    return text

def update_agent_list():
    all_agents = get_all_agents_from_crews() + st.session_state.new_agents
    with st.container(border=False):
        agent_cols = st.columns(5)
        for agent_index, agent in enumerate(all_agents):
            color_index = agent_index % len(agent_colors)
            with agent_cols[agent_index % 5]:
                card(
                    title=agent['role'],
                    text=f"Tools: {len(agent.get('tools', [])) or 'N/A'}",
                    styles=get_card_styles(color_index),
                    key=f"agent-card-{agent_index}-{random.randint(1, 100000)}"
                )
        with agent_cols[(len(all_agents)) % 5]:
            card(
                title="No Agents Added" if len(all_agents) == 0 else "",
                text="Click Here to Add an Agent" if len(all_agents) == 0 else "Click Here to Add Another Agent",
                on_click=show_agent_form,
                styles=get_empty_card_styles(),
                key=f"agent-card-add-new-{len(all_agents)}"
            )

        if st.session_state.show_agent_form:
            create_new_agent_form()

def update_task_list():
    all_tasks = get_all_tasks_from_crews() + st.session_state.new_tasks
    with st.container(border=False):
        task_cols = st.columns(5)
        for task_index, task in enumerate(all_tasks):
            color_index = task_index % len(agent_colors)
            agent_role = task.get('agent_role', 'No agent role')
            if 'agent_index' in task and 0 <= task['agent_index'] < len(st.session_state.new_agents):
                agent_role = st.session_state.new_agents[task['agent_index']]['role']

            truncated_description = truncate_text(task.get('description', 'No description'), 10)
            expected_output = truncate_text(task.get('expected_output', 'No expected output'), 4)

            with task_cols[task_index % 5]:
                card(
                    title=truncated_description,
                    text=[
                        f"Assigned Agent: {agent_role}",
                        f"Tools: {len(task.get('tools', []))}",
                        f"Expected Output: {expected_output}"
                    ],
                    styles=get_card_styles(color_index),
                    key=f"task-card-{task_index}-{random.randint(1, 100000)}"
                )
        with task_cols[(len(all_tasks)) % 5]:
            card(
                title="No Tasks Added" if len(all_tasks) == 0 else "",
                text="Click Here to Add a Task" if len(all_tasks) == 0 else "Click Here to Add Another Task",
                on_click=show_task_form,
                styles=get_empty_card_styles(),
                key=f"task-card-add-new-{len(all_tasks)}"
            )
        if st.session_state.show_task_form:
            create_new_task_form()

@st.experimental_fragment
def create_new_crew_container():
    with st.container(border=True):
        st.header("Create New Crew")
        tab1, tab2, tab3, tab4 = st.tabs(["Tools", "Agents", "Tasks", "Crew"])

        with tab1:
            st.subheader("Tools")
            tools_data = {
                "Active": [tool in st.session_state.active_tools for tool in TOOL_NAMES],
                "Tool": TOOL_NAMES,
                "Source": [tool["source"] for tool in TOOLS],
                "Description": [tool["description"] for tool in TOOLS],
                "Req. API Key": [tool["needsApiKey"] for tool in TOOLS]
            }
            tools_df = pd.DataFrame(tools_data)
            edited_tools_df = st.data_editor(tools_df, height=777, hide_index=True, 
                                            column_config={
                "Active": st.column_config.CheckboxColumn("Active", width=100),
                "Tool": st.column_config.TextColumn("Tool", disabled=True, width=300),
                "Source": st.column_config.TextColumn("Source", disabled=True, width=100),
                "Description": st.column_config.TextColumn("Description", disabled=True, width=1000),
                "Req. API Key": st.column_config.CheckboxColumn("Req. API Key", disabled=True, width=100)
            }, num_rows="fixed", key="tools_editor")

            if "tools_editor" in st.session_state:
                new_active_tools = edited_tools_df[edited_tools_df["Active"]]["Tool"].tolist()
                if set(new_active_tools) != set(st.session_state.active_tools):
                    st.session_state.active_tools = new_active_tools
                    save_user_preferences()
                    st.rerun()

        with tab2:
            st.subheader("Agents")
            update_agent_list()

        with tab3:
            st.subheader("Tasks")
            all_agents = get_all_agents_from_crews() + st.session_state.new_agents
            if not all_agents:
                st.warning("Please add an agent before creating a task.")
            else:
                update_task_list()

        with tab4:
            all_agents = get_all_agents_from_crews() + st.session_state.new_agents
            all_tasks = get_all_tasks_from_crews() + st.session_state.new_tasks

            with st.form(key="crew_form", clear_on_submit=True, border=False):
                crew_name = st.text_input("Crew Name", placeholder="Business Requirements Crew")

                # Populate the multi-selects with all agents and tasks
                selected_agents = st.multiselect("Add Agents", all_agents, format_func=lambda agent: agent['role'], key="crew_agents_multiselect", default=st.session_state.new_agents)
                selected_tasks = st.multiselect("Add Tasks", all_tasks, format_func=lambda task: task['description'], key="crew_tasks_multiselect", default=st.session_state.new_tasks)

                memory = st.toggle("Memory", key="crew_memory_checkbox", value=True)

                if st.form_submit_button("Create Crew"):
                    if not crew_name:
                        st.warning("Please enter a crew name before creating a crew.", icon="⚠️")
                    elif not selected_agents:
                        st.warning("Please add at least one agent before creating a crew.", icon="⚠️")
                    elif not selected_tasks:
                        st.warning("Please add at least one task before creating a crew.", icon="⚠️")
                    else:
                        warning_message = None
                        selected_agent_roles = [agent['role'] for agent in selected_agents]

                        for task in selected_tasks:
                            if 'agent_index' in task and task['agent_index'] < len(all_agents):
                                task_agent_role = all_agents[task['agent_index']]['role']
                                if task_agent_role not in selected_agent_roles:
                                    warning_message = f"Task {task['description']} has the agent {task_agent_role} assigned. Please remove the task or add the agent."
                                    break

                        if warning_message:
                            st.warning(warning_message, icon="⚠️")
                        else:
                            crew_data = {"name": crew_name, "agents": selected_agents, "tasks": selected_tasks, "memory": memory}
                            st.session_state.crew_list.append(crew_data)
                            update_crew_json(crew_data, len(st.session_state.crew_list) - 1)
                            for key in ['show_crew_container', 'show_agent_form', 'show_task_form']:
                                st.session_state[key] = False
                            st.rerun()

#endregion

#region Main Execution
if st.session_state.show_crew_container:
    create_new_crew_container()
else:
    handle_document_upload(st.session_state.langchain_upload_docs_selected)

    for msg in chat_messages_history.messages:
        if msg.type == "ai" and "[DEBUG]:" in msg.content:
            crew_name_match = re.findall(r"\[crewai_expander\]:==\s*(.*?)(?=\s|Interaction Logs|$)", msg.content, re.IGNORECASE)
            crew_start_time = re.findall(r"\[crewai_starttime\]:==\s*(.*?)(?:\s|$)", msg.content, re.IGNORECASE)
            crew_end_time = re.findall(r"\[crewai_endtime\]:==\s*(.*?)(?:\s|$)", msg.content, re.IGNORECASE)
            if crew_name_match:
                crew_name = crew_name_match[0]
                with st.chat_message("assistant"):
                    with st.expander(f"{crew_name} Interaction Logs", expanded=False):
                        st.write(msg.content)
                        crew_info = next((crew for crew in st.session_state.crew_list if crew['name'] == crew_name), None)
                        if crew_info:
                            agent_df = pd.DataFrame([agent['role'] for agent in crew_info['agents']], columns=["Agents"])
                            st.dataframe(agent_df, hide_index=True)
                            task_df = pd.DataFrame([task['description'] for task in crew_info['tasks']], columns=["Tasks"])
                            st.dataframe(task_df, hide_index=True)
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

        if any(st.session_state.crewai_crew_selected):
        # region Run CrewAi
            all_crews_finished = False
            for i, selected in enumerate(st.session_state.crewai_crew_selected):
                if selected:
                    start_time = time.time()
                    st.session_state[f"{st.session_state.crew_list[i]['name']}_start_time"] = start_time
                    crew_name = st.session_state.crew_list[i]['name']
                    st.session_state.current_crew_name = crew_name
                    agents = st.session_state.crew_list[i]['agents']
                    crew_context = json.dumps(st.session_state.crew_list[i])

                    # Update active tools based on the current crew
                    crew = st.session_state.crew_list[i]
                    st.session_state.crew_active_tools = set()
                    for agent in crew["agents"]:
                        for tool in agent.get("tools", []):
                            st.session_state.crew_active_tools.add(tool)
                    for task in crew["tasks"]:
                        for tool in task.get("tools", []):
                            st.session_state.crew_active_tools.add(tool)

                    # Check if the user's input satisfies the crew's tasks requirements
                    get_response(st.session_state.llm, user_input, "", chat_messages_history, context=crew_context)
                    
                    if st.session_state.can_run_crew:
                        with st.chat_message("assistant"):
                            with st.spinner(f"{crew_name} Working..."):
                                with st.expander(f"{crew_name} Interaction Logs", False):
                                    sys.stdout = StreamToExpander(st, crew_name, agents)

                                    dynamic_crew_handler = DynamicCrewHandler(
                                        name=crew_name,
                                        memory=st.session_state.crew_list[i]["memory"],
                                        agents=st.session_state.crew_list[i]["agents"],
                                        tasks=st.session_state.crew_list[i]["tasks"],
                                        llm=st.session_state.llm,
                                        user_prompt=user_input,
                                        chat_history=chat_messages_history,
                                        tools=[tool for tool in st.session_state.crew_active_tools]
                                    )

                                    response, new_crew_data = dynamic_crew_handler.create_crew()
                                    agent_df = pd.DataFrame([agent['role'] for agent in st.session_state.crew_list[i]['agents']], columns=["Agents"])
                                    st.dataframe(agent_df, hide_index=True)
                                    task_df = pd.DataFrame([task['description'] for task in st.session_state.crew_list[i]['tasks']], columns=["Tasks"])
                                    st.dataframe(task_df, hide_index=True)

                                    elapsed_time = time.time() - start_time
                                    st.empty().text(f"Total Time Elapsed: {elapsed_time:.2f} seconds")
                                    all_crews_finished = True
                        if not st.session_state.langchain_export_pdf_selected:  # We will be summarizing the response in the download PDF section
                            with st.chat_message("assistant"):
                                response_text = response[0] if isinstance(response, tuple) else response
                                st.write(response_text)
                                chat_messages_history.add_ai_message(response_text)
                                save_chat_history()

            if all_crews_finished:
                for i in range(len(st.session_state.crewai_crew_selected)):
                    if st.session_state.crewai_crew_selected[i]:
                        toggle_key = f"crew_{i}_selected"
                        st.session_state[toggle_key] = False

            # endregion
        else:
            # region Primary LLM
            if st.session_state.langchain_upload_docs_selected and st.session_state.vectorstore:
                if st.session_state.qa_chain:
                    response = st.session_state.qa_chain.invoke({"input": user_input})
                    with st.chat_message("assistant"):
                        st.write(response.get('answer', "No documents found, please upload documents."))
                        chat_messages_history.add_ai_message(response.get('answer', "No documents found, please upload documents."))
                        save_chat_history()
                        with st.expander("Sources", False):
                            for source_info in {f"{doc.metadata['source']}" for doc in response.get('context', []) if 'source' in doc.metadata}:
                                st.write(f"- {source_info}")
                else:
                    st.error("QA Chain is not initialized. Please check the configuration.")
            else:
                get_response(st.session_state.llm, user_input, "", chat_messages_history, context="")
                save_chat_history()

            # endregion

        # Export PDF. This is put at the end to get the full context of current chat history.
        if st.session_state.langchain_export_pdf_selected:
            get_response(st.session_state.llm, user_input, "export_pdf", chat_messages_history, context="")
            save_chat_history()

# endregion

#region CrewAi Edit Forms
edit_crew_agent_container = st.empty()
edit_crew_task_container = st.empty()

@st.experimental_fragment
def edit_crew_agent(agent, agent_index, crew_index):
    edit_crew_agent_container.empty()
    edit_crew_task_container.empty()
    current_llm = st.session_state.current_llm
    llm_index = llm_options.index(current_llm)
    with edit_crew_agent_container.container(border=True):
        st.header(f"Edit Agent {agent['role']}")

        if crew_index is not None:
            crew_name = st.text_input("Crew Name", value=st.session_state.crew_list[crew_index]["name"], placeholder="Business Requirements Crew", key=f"crew_agent_name_{agent_index}_{crew_index}")
            memory = st.toggle("Memory", value=st.session_state.crew_list[crew_index].get("memory", True), key=f"crew_agent_memory_{agent_index}_{crew_index}")
        role = st.text_input("Agent Name", value=agent["role"], key=f"agent_role_{agent_index}")
        tools = st.multiselect("Tools", TOOL_NAMES, default=agent.get("tools", []), key=f"agent_tools_{agent_index}")
        goal = st.text_area("Goal", value=agent["goal"], key=f"agent_goal_{agent_index}")
        backstory = st.text_area("Backstory", value=agent["backstory"], key=f"agent_backstory_{agent_index}")
        llm = st.selectbox("LLM", llm_options, index=llm_index, key=f"agent_llm_{agent_index}", disabled=True)
        allow_delegation = st.toggle("Allow Delegation", value=agent.get("allow_delegation", False), key=f"agent_allow_delegation_{agent_index}")

        if st.button(label="Save Agent", key=f"save_agent_{agent_index}", type="primary"):
            st.session_state.rerun_needed = True
            if crew_index is not None:
                st.session_state.crew_list[crew_index].update({
                    "name": crew_name,
                    "memory": memory
                })
            st.session_state.crew_list[crew_index]["agents"][agent_index] = {
                "role": role, "goal": goal, "backstory": backstory, "llm": llm,
                "allow_delegation": allow_delegation, "tools": tools
            }
            update_crew_json(st.session_state.crew_list[crew_index], crew_index)
            edit_crew_agent_container.empty()
            st.rerun()
        if st.button(label="Cancel", key=f"cancel_agent_{agent_index}"):
            edit_crew_task_container.empty()
            edit_crew_agent_container.empty()

@st.experimental_fragment
def edit_crew_task(task, task_index, crew_index):
    edit_crew_task_container.empty()
    edit_crew_agent_container.empty()
    crew = next((crew for crew in st.session_state.crew_list if task in crew["tasks"]), None)

    current_agents = crew["agents"]
    agent_names = [agent['role'] for agent in current_agents]
    
    # Get active tools
    active_tools = st.session_state.active_tools

    # Create the context task options excluding the current task and only including tasks below the current task
    context_task_options = [f'Task {i+1}' for i in range(task_index)]
    default_context_tasks = [f'Task {i+1}' for i in task.get("context_indexes", []) if i < task_index]

    with edit_crew_task_container.container(border=True):
        st.header(f"Edit Task {task_index + 1}")
        crew_name = st.text_input("Crew Name", value=st.session_state.crew_list[crew_index]["name"], placeholder="Business Requirements Crew", key=f"crew_task_name_{task_index}_{crew_index}")
        memory = st.toggle("Memory", value=st.session_state.crew_list[crew_index].get("memory", True), key=f"crew_task_memory_{task_index}_{crew_index}")
        description = st.text_area("Description", value=task["description"], key=f"task_description_{task_index}")
        tools = st.multiselect("Tools", active_tools, default=task.get("tools", []), key=f"task_tools_{task_index}")
        assigned_agent = st.selectbox("Assigned Agent", agent_names, index=task["agent_index"], key=f"task_agent_{task_index}")
        expected_output = st.text_area("Expected Output", value=task.get("expected_output", ""), key=f"task_expected_output_{task_index}")

        # Conditional logic for context task selection
        if context_task_options:
            context_indexes = st.multiselect(
                "Add previous tasks to context",
                options=context_task_options,
                default=default_context_tasks,
                key=f"task_context_indexes_{task_index}"
            )
        else:
            st.text_input("Add previous tasks to context", value="N/A", disabled=True)

        if st.button(label="Save Task", key=f"save_task_{task_index}", type="primary"):
            st.session_state.rerun_needed = True
            st.session_state.crew_list[crew_index].update({
                "name": crew_name,
                "memory": memory
            })
            st.session_state.crew_list[crew_index]["tasks"][task_index] = {
                "description": description,
                "agent_index": agent_names.index(assigned_agent),  # Ensure this is an integer index
                "expected_output": expected_output,
                "context_indexes": [int(idx.split()[-1]) - 1 for idx in context_indexes] if context_task_options else [],
                "tools": tools
            }
            update_crew_json(st.session_state.crew_list[crew_index], crew_index)
            edit_crew_task_container.empty()
            st.rerun()
        if st.button(label="Cancel", key=f"cancel_task_{task_index}"):
            edit_crew_task_container.empty()
            edit_crew_agent_container.empty()

if st.session_state.rerun_needed:
    st.session_state.rerun_needed = False
    st.rerun() 

#endregion

#region CrewAi Sidebar List
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
                        truncated_role = truncate_text(agent['role'], 10)
                        with agent_col1:
                            card(
                                title=truncated_role,
                                text=[
                                    f"Tools: {len(agent.get('tools', [])) or 'N/A'}"
                                ],
                                on_click=lambda idx=agent_index, cidx=i: (edit_crew_agent(agent, idx, cidx)),
                                styles=get_card_styles(color_index),
                                key=f"agent-card-{i}-{agent_index}"
                            )

                    tasks_by_agent = sorted(crew["tasks"], key=lambda x: x['agent_index'])
                    for task_index, task in enumerate(tasks_by_agent):
                        task_agent_role = crew['agents'][task['agent_index']]['role'] if 0 <= task['agent_index'] < len(crew['agents']) else 'Invalid Agent Index'
                        color_index = task['agent_index'] % len(agent_colors)

                        truncated_description = truncate_text(task.get('description', 'No description'), 10)
                        expected_output = truncate_text(task.get('expected_output', 'No expected output'), 4)

                        with task_col2:
                            card(
                                title=truncated_description,
                                text=[
                                    f"Assigned Agent: {task_agent_role}",
                                    f"Tools: {len(task.get('tools', []))}",
                                    f"Expected Output: {expected_output}"
                                ],
                                on_click=lambda tidx=task_index, cidx=i: (edit_crew_task(task, tidx, cidx)),
                                styles=get_card_styles(color_index),
                                key=f"task-card-{i}-{task_index}"
                            )

                    if st.button("Delete Crew", key=f"delete_crew_{i}"):
                        delete_crew(i)
                        st.rerun()

display_crew_list()

#endregion