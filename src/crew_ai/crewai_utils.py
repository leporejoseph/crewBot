# src/crew_ai/crewai_utils.py

import os
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

class DynamicCrewHandler:
    def __init__(self, name, agents, tasks, llm, user_prompt, chat_history):
        self.name = name
        self.agents = agents
        self.tasks = tasks
        self.llm = llm
        self.user_prompt = user_prompt
        self.chat_history = chat_history

    def get_tool_instance(self, tool_name):
        tools = {
            "SerperDevTool": SerperDevTool(api_key=os.getenv("SERPER_API_KEY")),
            "BrowserbaseLoadTool": BrowserbaseLoadTool(api_key=os.getenv("BROWSERBASE_API_KEY")),
            "ScrapeWebsiteTool": ScrapeWebsiteTool(),
            "DirectoryReadTool": DirectoryReadTool(),
            "FileReadTool": FileReadTool(),
            "SeleniumScrapingTool": SeleniumScrapingTool(),
            "DirectorySearchTool": DirectorySearchTool(),
            "PDFSearchTool": PDFSearchTool(),
            "TXTSearchTool": TXTSearchTool(),
            "CSVSearchTool": CSVSearchTool(),
            "XMLSearchTool": XMLSearchTool(),
            "JSONSearchTool": JSONSearchTool(),
            "DOCXSearchTool": DOCXSearchTool(),
            "MDXSearchTool": MDXSearchTool(),
            "PGSearchTool": PGSearchTool(),
            "WebsiteSearchTool": WebsiteSearchTool(),
            "GithubSearchTool": GithubSearchTool(),
            "CodeDocsSearchTool": CodeDocsSearchTool(),
            "YoutubeVideoSearchTool": YoutubeVideoSearchTool(),
            "YoutubeChannelSearchTool": YoutubeChannelSearchTool()
        }
        return tools.get(tool_name, None)

    def create_agents(self):
        agents = []
        for i, agent in enumerate(self.agents):
            tools = [self.get_tool_instance(tool) for tool in agent.get("tools", [])]
            goal = agent['goal']
            if i == 0:
                goal += f"\n\nUse user prompt and Chat history for context.\n\n[Start of user prompt]\n{self.user_prompt}\n[End of user prompt]\n\n[Start of chat history]\n{self.chat_history}\n[End of chat history]"
            agents.append(Agent(
                role=agent["role"],
                goal=goal,
                backstory=agent["backstory"],
                llm=self.llm,
                allow_delegation=agent["allow_delegation"],
                memory=agent["memory"],
                tools=tools
            ))
        return agents

    def create_tasks(self, agent_objects):
        tasks = []
        for task in self.tasks:
            agent = agent_objects[task["agent_index"]]
            context_tasks = [tasks[idx] for idx in task.get("context_indexes", [])]
            tools = [self.get_tool_instance(tool) for tool in task.get("tools", [])]
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
            "agents": self.agents,
            "tasks": self.tasks
        }
        return response, new_crew_data

def create_agent(role, goal, backstory, llm, user_prompt=None, chat_history=None):
    if user_prompt and chat_history:
        goal += f"\n\nUse user prompt and Chat history for context.\n\n[Start of user prompt]\n{user_prompt}\n[End of user prompt]\n\n[Start of chat history]\n{chat_history}\n[End of chat history]"
    return Agent(role=role, goal=goal, backstory=backstory, llm=llm, allow_delegation=False, memory=True)

def create_task(description, agent, expected_output, context_indexes=[]):
    return Task(description=description, agent=agent, expected_output=expected_output, context=[context_indexes])