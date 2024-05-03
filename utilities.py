import re
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

class StreamToExpander:
    def __init__(self, expander, agent_task_outputs, task_values):
        self.expander = expander
        self.buffer = []
        self.agent_colors = {
            "Market Research Analyst": "blue",
            "Technology Expert": "red",
            "Business Development Consultant": "green",
            "Summary Agent": "violet",
            "Project Manager": "orange",
        }
        self.color_index = 0
        self.agent_task_outputs = agent_task_outputs
        self.task_values = task_values

    def write(self, data):
        # Filter ANSI escape codes
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Match agent name and task output
        agent_name_match = re.search(r"== Working Agent: (.*?)\n", cleaned_data)
        task_output_match = re.search(r"== \[(.*?)\] Task output: (.*?)\n", cleaned_data, re.DOTALL)

        if agent_name_match and task_output_match:
            agent_name = agent_name_match.group(1)
            task_output = task_output_match.group(2)

            # Append to the given lists
            self.agent_task_outputs.append({"Agent": agent_name, "Output": task_output})
            st.toast(":robot_face: " + agent_name)

        # Replace agent names with color tags
        for agent_name, color in self.agent_colors.items():
            cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")

        # Extract task-related information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            self.task_values.append(f"Task: {task_value}")

        self.buffer.append(cleaned_data)

        # Display output when there's a newline
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


def get_response(llm, user_query, chat_history):
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
