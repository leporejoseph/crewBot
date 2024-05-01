import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import sys
import time
from crewai import Agent, Task, Crew, Process
from langchain import hub
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain.agents import AgentExecutor, tool
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from typing import List, Dict
from pydantic import BaseModel, Field
from streamlit_chat import message as chat_message

import json
import os
import re

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    # Enter in your OPENAI API KEY in the .env file if using OpenAi
    api_key = os.getenv("OPENAI_API_KEY"),

    # Configure local LLM using Ollama and LM Studio if using Local
    # Steps:
        # Download Ollama - https://ollama.com/
        # Download LM Studio - https://lmstudio.ai/
            # In LM Studio, download an LLM model and run local server 
            # QuantFactory/dolphin-2.9-llama3-8b-GGUF model seemed to work as of 5/1/24
        # Uncomment 3 lines below and enter in model name found in LM Studio
    # model='ENTER IN MODEL NAME HERE',
    # base_url="http://localhost:1234/v1", 
    # api_key="NA",
    streaming=True
)

# List to store agent outputs
agent_task_outputs = []

# Colors mapping
colors = ['red', 'green', 'blue', 'orange', 'purple', 'pink', 'yellow', 'teal']

# Class to handle console output and display in Streamlit with color changes
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.agent_colors = {
            "Market Research Analyst": "blue",
            "Technology Expert": "red",
            "Business Development Consultant": "green",
            "Summary Agent": "blue",
            "Project Manager": "orange",
        }

        self.color_index = 0

    def write(self, data):
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Extract the agent's name and output
        agent_name_match = re.search(r"== Working Agent: (.*?)\n", cleaned_data)
        task_output_match = re.search(r"== \[(.*?)\] Task output: (.*?)\n", cleaned_data, re.DOTALL)

        if agent_name_match and task_output_match:
            agent_name = agent_name_match.group(1)
            task_output = task_output_match.group(2)

            # Store in the global list
            agent_task_outputs.append({"Agent": agent_name, "Output": task_output})

            # Display toast with agent name
            st.toast(":robot_face: " + agent_name)

        for agent_name, color in self.agent_colors.items():
            if agent_name in cleaned_data:
                cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")

        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            task_values.append(f"Task: {task_value}")

        self.buffer.append(cleaned_data)

        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Define the agents for the crew
def create_crewai_setup(chat_history, user_query):
    # Define the agents
    market_research_analyst = Agent(
        role="Market Research Analyst",
        goal=f"Analyze the market demand and suggest marketing strategies.",
        backstory="A seasoned professional with a deep understanding of market dynamics, target audiences, and competitors.",
        llm=llm,
        allow_delegation=False,
    )

    technology_expert = Agent(
        role="Technology Expert",
        goal=f"Assess technological feasibilities and requirements for producing high-quality reports.",
        backstory="A visionary in technological trends, identifying which technologies best suit the project's needs.",
        llm=llm,
        allow_delegation=False,
    )

    business_development_consultant = Agent(
        role="Business Development Consultant",
        goal=f"Evaluate the business model, focusing on scalability and revenue streams reports.",
        backstory="Seasoned in shaping business strategies, ensuring long-term sustainability for products.",
        llm=llm,
        allow_delegation=False,
    )

    project_manager = Agent(
        role="Project Manager",
        goal="Coordinate between teams, track progress, and manage scope.",
        backstory="A skilled manager ensuring projects run smoothly and on time.",
        llm=llm,
        allow_delegation=False,
    )

    # Summary agent
    summary_agent = Agent(
        role="Summary Agent",
        goal=f"Compile a comprehensive report summarizing all agents' tasks and outputs into a cohesive document based Use the context to fill out the template below: [start of JSON context] {json.dumps(agent_task_outputs)} [end of JSON context]",
        backstory="A dedicated summarizer skilled in creating cohesive narratives.",
        llm=llm,
        allow_delegation=False,
    )

    # Define tasks
    market_research_analyst_task = Task(
        description=f"""
                Analyze the market demand for the app, write a report on the ideal customer profile, and suggest marketing strategies. 
                Show 3 potential competitors and pros and cons for each based on the most recent prompt: 
                --Start of prompt""
                {user_query} 
                --End of prompt-- 
                --Chat history--
                {chat_history}
                --End of chat history--
            """,
        agent=market_research_analyst,
        expected_output="Market analysis report. IMPORTANT: Use bullet points.",
    )

    technology_expert_task = Task(
        description="Assess the technological aspects of the app's development, write a report on necessary technologies and approaches. Use the most up-to-date tools for optimized fast-track development.",
        agent=technology_expert,
        expected_output="Technological assessment report. IMPORTANT: Use bullet points. Make sure to output the software architecture",
        context=[market_research_analyst_task]
    )

    business_development_consultant_task = Task(
        description="Evaluate the business model, focusing on scalability and revenue streams, and create a business plan.",
        agent=business_development_consultant,
        expected_output="Business model evaluation report. IMPORTANT: Use bullet points.",
        context=[
            market_research_analyst_task, 
            technology_expert_task]
    )

    project_manager_task = Task(
        description="Identify and pair teams, manage project scope, and create a timeline for each section.",
        agent=project_manager,
        expected_output="Project management plan that breaks down the roles for each stage of development from the context. IMPORTANT: Use bullet points.",
        context=[
            market_research_analyst_task, 
            technology_expert_task, 
            business_development_consultant_task]
    )

    # Summary task
    summary_task = Task(
    description=f"""
            Use all given context from prior agents to compile the template provided in the expected output based on the most 
            recent prompt: {user_query} --End of prompt-- --Chat history--{chat_history} 
            IMPORTANT: Use template and fill out the bottom section 
            Market Research: Very detailed bullet list from market research analyst Results.
            Technology Expert: Very detailed bullet list from technology expert Results.
            Business Development Consultant: Very detailed bullet list from business development consultant Results.
            Project Manager: Very detailed bullet list from project manager Results.
        """,
    agent=summary_agent,
    expected_output=f"""
            [template that must be used]
            ### Business Requirement Document
            1. **Introduction:**
                - Purpose: Describes the document's purpose and scope.
                - Scope: Defines project boundaries.
                - Objectives: Outlines key goals.
                - Assumptions: Lists assumptions made.
                - Constraints: Highlights limitations.

            2. **Project Overview:**
                - Background: Context on project inception.
                - Goals: Lists primary goals.
                - Deliverables: Defines key outputs.
                - Milestones: Significant milestones.

            3. **Stakeholders:**
                - List: Name, role, interest, responsibilities.

            4. **Business Requirements:**
                - Needs: ID, description, priority, rationale from the business consultant's report.

            5. **Functional Requirements:**
                - Features: ID, description, priority, and acceptance criteria.

            6. **Non-Functional Requirements:**
                - Attributes: ID, description, type, and acceptance criteria.

            7. **Technical Architecture:**
                - Overview: Architecture overview derived from the technological expert's report.
                - Components: Name, description.

            8. **Implementation Plan:**
                - Strategy: Implementation strategy.
                - Timeline: Phase, description, duration.
                - Resources: Defines necessary resources.

            9. **Risk Management:**
                - Risks: ID, description, likelihood, impact, and mitigation strategies.

            10. **Conclusion:**
                - Summary: Reiterates key points from all agents.

            11. **Appendices:**
                - Market Research: Give me a bullet list from market research analyst results that you have gathered.
                - Technology Expert: Give me a  detailed bullet list from technology expert results that you have gathered.
                - Business Development Consultant: Give me a  detailed bullet list from business development consultant results that you have gathered.
                - Project Manager: Give me a  detailed bullet list from project manager results that you have gathered.
                - Give me additional information related to the project.
            

            [end of template]
        """,
        context=[
            market_research_analyst_task, 
            technology_expert_task, 
            business_development_consultant_task, 
            project_manager_task]
    )


    # Form the crew
    crew = Crew(
        agents=[
            market_research_analyst, 
            technology_expert, 
            business_development_consultant,
            project_manager, 
            summary_agent],
        tasks=[
            market_research_analyst_task, 
            technology_expert_task, 
            business_development_consultant_task, 
            project_manager_task, 
            summary_task],
        process=Process.sequential,
        verbose=True,
    )

    return crew.kickoff()

# App config and Streamlit setup
st.title("Crewbot: Your AI Assistant")
st.sidebar.title("Configuration")
run_agents = st.sidebar.checkbox("Run Business Analysis", value=False)
st.sidebar.subheader("About the Team:")
st.sidebar.markdown("**Business Analysis** - Enter in product name and let AI do an analysis.")

# Keep track of performed tasks
task_values = []

# Function for user response
def get_response(user_query, chat_history):
    template = f"""
    You are a helpful assistant. Answer the following questions considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# Session state handling
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Conversation display
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("What's on your mind?")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        if run_agents:
            start_time = time.time()
            with st.spinner("Agents Working..."):
                    with st.expander("Crew Interaction Logs", False):
                        sys.stdout = StreamToExpander(st)
                        response = create_crewai_setup(user_query, st.session_state.chat_history)
            
            stopwatch_placeholder = st.empty()

            end_time = time.time()
            total_time = end_time - start_time
            stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

            st.header("Tasks:")
            st.table({"Tasks": task_values})

            st.header("Results:")
            st.markdown(response)
            st.session_state.messages.append({"role": "AI", "content": response})

        else:
            response = st.write_stream(get_response(user_query, st.session_state.chat_history))
            st.session_state.messages.append({"role": "AI", "content": response})

    st.session_state.chat_history.append(AIMessage(content=response))