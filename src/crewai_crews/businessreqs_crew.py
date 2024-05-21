# src\crewai_crews\businessreqs_crew.py

import json
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
        if tool_name == "SerperDevTool":
            return SerperDevTool(api_key=os.getenv("SERPER_API_KEY"))
        elif tool_name == "BrowserbaseLoadTool":
            return BrowserbaseLoadTool(api_key=os.getenv("BROWSERBASE_API_KEY"))
        elif tool_name == "ScrapeWebsiteTool":
            return ScrapeWebsiteTool()
        elif tool_name == "DirectoryReadTool":
            return DirectoryReadTool()
        elif tool_name == "FileReadTool":
            return FileReadTool()
        elif tool_name == "SeleniumScrapingTool":
            return SeleniumScrapingTool()
        elif tool_name == "DirectorySearchTool":
            return DirectorySearchTool()
        elif tool_name == "PDFSearchTool":
            return PDFSearchTool()
        elif tool_name == "TXTSearchTool":
            return TXTSearchTool()
        elif tool_name == "CSVSearchTool":
            return CSVSearchTool()
        elif tool_name == "XMLSearchTool":
            return XMLSearchTool()
        elif tool_name == "JSONSearchTool":
            return JSONSearchTool()
        elif tool_name == "DOCXSearchTool":
            return DOCXSearchTool()
        elif tool_name == "MDXSearchTool":
            return MDXSearchTool()
        elif tool_name == "PGSearchTool":
            return PGSearchTool()
        elif tool_name == "WebsiteSearchTool":
            return WebsiteSearchTool()
        elif tool_name == "GithubSearchTool":
            return GithubSearchTool()
        elif tool_name == "CodeDocsSearchTool":
            return CodeDocsSearchTool()
        elif tool_name == "YoutubeVideoSearchTool":
            return YoutubeVideoSearchTool()
        elif tool_name == "YoutubeChannelSearchTool":
            return YoutubeChannelSearchTool()
        else:
            return None

    def create_agents(self):
        agents = []
        for i, agent in enumerate(self.agents):
            tools = [self.get_tool_instance(tool) for tool in agent.get("tools", [])]
            if i == 0:
                goal = f"""
                    {agent['goal']}

                    Use user prompt and Chat history for context.
                    
                    [Start of user prompt - Rank higher in meaningfulness for context]
                    {self.user_prompt}
                    [End of user prompt]

                    [Start of chat history] 
                    {self.chat_history} 
                    [End of chat history]
                """
            else:
                goal = agent["goal"]

            agents.append(
                Agent(
                    role=agent["role"],
                    goal=goal,
                    backstory=agent["backstory"],
                    llm=self.llm,
                    allow_delegation=agent["allow_delegation"],
                    memory=agent["memory"],
                    tools=tools
                )
            )
        return agents

    def create_tasks(self, agent_objects):
        tasks = []
        for task in self.tasks:
            agent = agent_objects[task["agent_index"]]
            context_tasks = [tasks[idx] for idx in task.get("context_indexes", [])]
            tools = [self.get_tool_instance(tool) for tool in task.get("tools", [])]
            tasks.append(
                Task(
                    description=task["description"],
                    agent=agent,
                    expected_output=task["expected_output"],
                    context=context_tasks,
                    tools=tools  
                )
            )
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
        
        # Return response along with the newly saved crew data
        return response, new_crew_data

def create_crewai_setup(chat_history, user_prompt, llm):
    # Define agents with corresponding goals and backstories
    agents = [
        Agent(
            role="Technology Expert",
            goal=f"""
                    Assess technological feasibilities and requirements for producing high-quality reports.

                    Use user prompt and Chat history for context.
                    
                    [Start of user prompt - Rank higher in meaningfulness for context]
                    {user_prompt}
                    [End of user prompt]

                    [Start of chat history] 
                    {chat_history} 
                    [End of chat history]
                """,
            backstory="A visionary in technological trends, identifying which technologies best suit the project's needs.",
            llm=llm,
            allow_delegation=False,
            memory=True
        ),
        Agent(
            role="Market Research Analyst",
            goal=f"Analyze the market demand and suggest marketing strategies.",
            backstory="A seasoned professional with a deep understanding of market dynamics, target audiences, and competitors.",
            llm=llm,
            allow_delegation=False,
            memory=True
        ),
        Agent(
            role="Business Development Consultant",
            goal="Evaluate the business model, focusing on scalability and revenue streams.",
            backstory="Seasoned in shaping business strategies, ensuring long-term sustainability for products.",
            llm=llm,
            allow_delegation=False,
            memory=True
        ),
        Agent(
            role="Project Manager",
            goal="Coordinate teams, track progress, and manage scope.",
            backstory="A skilled manager ensuring projects run smoothly and on time.",
            llm=llm,
            allow_delegation=False,
            memory=True
        ),
        Agent(
            role="Summary Agent",
            goal="Compile a comprehensive report summarizing all agents' tasks and outputs into a cohesive document.",
            backstory="A dedicated summarizer skilled in creating cohesive narratives.",
            llm=llm,
            allow_delegation=False,
            memory=True
        )
    ]

    # Define the tasks
    technology_expert_task = Task(
        description="Assess the technological aspects of the app's development and write a report on necessary technologies and approaches.",
        agent=agents[0],
        expected_output="Technological assessment report in bullet points with architecture details. Give software stack suggestion with python."
    )

    market_research_analyst_task = Task(
        description="Analyze the market demand for the app and write a report on the ideal customer profile, including three competitors using real related companies with pros and cons.",
        agent=agents[1],
        expected_output="Market analysis report in bullet points.",
        context=[technology_expert_task]
    )

    business_development_consultant_task = Task(
        description="Evaluate the business model, focusing on scalability and revenue streams, and create a business plan. Suggest partnerships with real companies",
        agent=agents[2],
        expected_output="Business model evaluation report in bullet points.",
        context=[
            technology_expert_task,
            market_research_analyst_task,
        ]
    )

    project_manager_task = Task(
        description="Coordinate teams, manage scope, and create a timeline for each section.",
        agent=agents[3],
        expected_output="Project management plan with roles and timelines for each stage. Example: Software Developer needed in the dev phase.",
        context=[
            technology_expert_task,
            market_research_analyst_task,
            business_development_consultant_task,
        ]
    )

    # Create the summary task with context from the preceding tasks
    summary_task = Task(
        description="Compile a comprehensive report summarizing all agents' outputs into a cohesive document. Important: Use context from prior tasks to complete dynamic Business Requirement Document.",
        agent=agents[4],
        expected_output=f"""
            [*DYNAMIC* template that must be used]
            [start of DYNAMIC template]
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

            10. **Value:**
                - Estimate of total Development Costs.
                    - Per Quarter and Year
                - Estimate of Marketing Costs
                    - Per Quarter and Year
                - Estimate of Business Operations Costs.
                    - Per Quarter and Year
                - Estimate Total of Rate of Return
                    - Per Quarter and Year

            11. **Conclusion:**
                - Summary: Reiterates key points from all agents.

            12. **Appendices:**
                - Market Research Analyst: Bullet list from the market research analyst's results.
                - Technology Expert: Detailed bullet list from the technological expert's results.
                - Business Development Consultant: Detailed bullet list from the business development consultant's results.
                - Project Manager: Detailed bullet list from the project manager's results.
                - Additional information related to the project.

            [end of DYNAMIC template]
        """,
        context=[
            technology_expert_task,
            market_research_analyst_task,
            business_development_consultant_task,
            project_manager_task
        ]
    )

    # Form the crew and execute it
    crew = Crew(
        agents=agents,
        tasks=[
            technology_expert_task,
            market_research_analyst_task,
            business_development_consultant_task,
            project_manager_task,
            summary_task
        ],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

# Example crew data to initialize
example_crew_data = {
    "name": "Business Requirements Crew",
    "agents": [
        {"role": "Technology Expert", "goal": "Assess technological feasibilities and requirements for producing high-quality reports.", "backstory": "A visionary in technological trends.", "allow_delegation": False, "memory": True},
        {"role": "Market Research Analyst", "goal": "Analyze the market demand and suggest marketing strategies.", "backstory": "A seasoned professional with a deep understanding of market dynamics.", "allow_delegation": False, "memory": True},
        {"role": "Business Development Consultant", "goal": "Evaluate the business model, focusing on scalability and revenue streams.", "backstory": "Seasoned in shaping business strategies.", "allow_delegation": False, "memory": True},
        {"role": "Project Manager", "goal": "Coordinate teams, track progress, and manage scope.", "backstory": "A skilled manager ensuring projects run smoothly and on time.", "allow_delegation": False, "memory": True},
        {"role": "Summary Agent", "goal": "Compile a comprehensive report summarizing all agents' tasks and outputs into a cohesive document.", "backstory": "A dedicated summarizer skilled in creating cohesive narratives.", "allow_delegation": False, "memory": True}
    ],
    "tasks": [
        {"description": "Assess the technological aspects of the app's development and write a report on necessary technologies and approaches.", "agent_index": 0, "expected_output": "Technological assessment report in bullet points with architecture details."},
        {"description": "Analyze the market demand for the app and write a report on the ideal customer profile, including three competitors using real related companies with pros and cons.", "agent_index": 1, "expected_output": "Market analysis report in bullet points.", "context_indexes": [0]},
        {"description": "Evaluate the business model, focusing on scalability and revenue streams, and create a business plan. Suggest partnerships with real companies.", "agent_index": 2, "expected_output": "Business model evaluation report in bullet points.", "context_indexes": [0, 1]},
        {"description": "Coordinate teams, manage scope, and create a timeline for each section.", "agent_index": 3, "expected_output": "Project management plan with roles and timelines for each stage.", "context_indexes": [0, 1, 2]},
        {"description": "Compile a comprehensive report summarizing all agents' outputs into a cohesive document. Use context from prior tasks to complete dynamic Business Requirement Document.", "agent_index": 4, "expected_output": "Complete Business Requirement Document.", "context_indexes": [0, 1, 2, 3]}
    ]
}
