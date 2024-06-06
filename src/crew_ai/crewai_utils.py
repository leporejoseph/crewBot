# src/crew_ai/crewai_utils.py

import os, json
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import (
    SerperDevTool, BrowserbaseLoadTool,
    ScrapeWebsiteTool, DirectoryReadTool, FileReadTool, SeleniumScrapingTool,
    DirectorySearchTool, PDFSearchTool, TXTSearchTool,
    CSVSearchTool, XMLSearchTool, JSONSearchTool,
    DOCXSearchTool, MDXSearchTool, PGSearchTool,
    WebsiteSearchTool, GithubSearchTool, CodeDocsSearchTool,
    YoutubeVideoSearchTool, YoutubeChannelSearchTool
)

# Load environment variables from .env file
load_dotenv()

TOOLS = [{"name": tool, "needsApiKey": False, "source": "crewai", "description": f"A RAG tool for {tool.lower().replace('searchtool', 'searching within ')}"} for tool in [
    "CSVSearchTool", "CodeDocsSearchTool", "DOCXSearchTool", "DirectoryReadTool", "DirectorySearchTool", "FileReadTool", "GithubSearchTool", "JSONSearchTool",
    "MDXSearchTool", "PDFSearchTool", "PGSearchTool", "RagTool", "ScrapeElementFromWebsiteTool", "ScrapeWebsiteTool", "SerperDevTool", "TXTSearchTool",
    "WebsiteSearchTool", "XMLSearchTool", "YoutubeChannelSearchTool", "YoutubeVideoSearchTool"]]

class DynamicCrewHandler:
    def __init__(self, name, memory, agents, tasks, llm, user_prompt, chat_history, tools=None):
        self.name = name
        self.memory = memory
        self.agents = agents
        self.tasks = tasks
        self.llm = llm
        self.user_prompt = user_prompt
        self.chat_history = chat_history
        self.tools = tools or []

    def get_tool_instance(self, tool_name):
        tools = {
            "SerperDevTool": lambda: SerperDevTool(api_key=os.getenv("SERPER_API_KEY")),
            "BrowserbaseLoadTool": lambda: BrowserbaseLoadTool(api_key=os.getenv("BROWSERBASE_API_KEY")),
            "ScrapeWebsiteTool": lambda: ScrapeWebsiteTool(),
            "DirectoryReadTool": lambda: DirectoryReadTool(),
            "FileReadTool": lambda: FileReadTool(),
            "SeleniumScrapingTool": lambda: SeleniumScrapingTool(),
            "DirectorySearchTool": lambda: DirectorySearchTool(),
            "PDFSearchTool": lambda: PDFSearchTool(),
            "TXTSearchTool": lambda: TXTSearchTool(),
            "CSVSearchTool": lambda: CSVSearchTool(),
            "XMLSearchTool": lambda: XMLSearchTool(),
            "JSONSearchTool": lambda: JSONSearchTool(),
            "DOCXSearchTool": lambda: DOCXSearchTool(),
            "MDXSearchTool": lambda: MDXSearchTool(),
            "PGSearchTool": lambda: PGSearchTool(table_name=""),
            "WebsiteSearchTool": lambda: WebsiteSearchTool(),
            "GithubSearchTool": lambda: GithubSearchTool(),
            "CodeDocsSearchTool": lambda: CodeDocsSearchTool(),
            "YoutubeVideoSearchTool": lambda: YoutubeVideoSearchTool(),
            "YoutubeChannelSearchTool": lambda: YoutubeChannelSearchTool()
        }
        
        if tool_name in st.session_state.crew_active_tools:
            return tools[tool_name]()
        return None

    def create_agents(self):
        agents = []
        for i, agent in enumerate(self.agents):
            tools = [self.get_tool_instance(tool) for tool in agent.get("tools", []) if self.get_tool_instance(tool) is not None]
            goal = agent['goal']
            if i == 0:
                goal += f"\n\nUse user prompt and Chat history for context.\n\n[Start of user prompt]\n{self.user_prompt}\n[End of user prompt]\n\n[Start of chat history]\n{self.chat_history}\n[End of chat history]"
            agents.append(Agent(
                role=agent["role"],
                goal=goal,
                backstory=agent["backstory"],
                llm=self.llm,
                allow_delegation=agent["allow_delegation"],
                tools=tools
            ))
        return agents

    def create_tasks(self, agent_objects):
        tasks = []
        for task in self.tasks:
            agent = agent_objects[task["agent_index"]]
            context_tasks = [tasks[idx] for idx in task.get("context_indexes", [])]
            tools = [self.get_tool_instance(tool) for tool in task.get("tools", []) if self.get_tool_instance(tool) is not None]
            tasks.append(Task(
                description=task["description"],
                agent=agent,
                expected_output=task["expected_output"],
                context=context_tasks,
                tools=tools
            ))
        return tasks

    def create_crew(self):
        agents = self.create_agents()
        tasks = self.create_tasks(agents)
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        response = crew.kickoff()
        new_crew_data = {
            "name": self.name,
            "memory": self.memory,
            "agents": self.agents,
            "tasks": self.tasks
        }
        return response, new_crew_data

# Crew Management
def update_crew_json(updated_crew, crew_index):
    """Update the specific crew in the crews.json file."""
    crews_file = 'crew_ai/crews.json'

    with open(crews_file, 'r') as file:
        crews_data = json.load(file)

    if 0 <= crew_index < len(crews_data):
        crews_data[crew_index] = updated_crew
    else:
        crews_data.append(updated_crew)
    
    with open(crews_file, 'w') as file:
        json.dump(crews_data, file, indent=4)

def delete_crew(index):
    """Delete a specific crew by index."""
    st.session_state.crew_list.pop(index)
    crews_file = 'crew_ai/crews.json'
    
    with open(crews_file, 'r') as file:
        crews_data = json.load(file)
    
    if 0 <= index < len(crews_data):
        crews_data.pop(index)
        with open(crews_file, 'w') as file:
            json.dump(crews_data, file, indent=4)
    else:
        st.error(f"Crew index {index} is out of range.")
