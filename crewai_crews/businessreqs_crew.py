import json
from crewai import Agent, Task, Crew, Process

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
        context=[ technology_expert_task ]
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
                - Market Research Analyst: Bullet list from the market research analyst's results.
                - Technology Expert: Detailed bullet list from the technological expert's results.
                - Business Development Consultant: Detailed bullet list from the business development consultant's results.
                - Project Manager: Detailed bullet list from the project manager's results.
                - Additional information related to the project.

            [end of template]
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
