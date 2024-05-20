# src/utils/streamlit_expander.py
import streamlit as st
import re

class StreamToExpander:
    """Redirect output to a Streamlit expander with formatted text and color coding for different agents."""
    def __init__(self, expander, agent_task_outputs, task_values):
        self.expander, self.agent_task_outputs, self.task_values, self.buffer = expander, agent_task_outputs, task_values, []
        self.agent_colors = {"Market Research Analyst": "blue", 
                             "Technology Expert": "red", 
                             "Business Development Consultant": "green", 
                             "Summary Agent": "violet", 
                             "Project Manager": "orange"}

    def write(self, data):
        """Process and format the input data, extract agent names and task outputs, and append to buffer for display."""
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        agent_name_match, task_output_match = re.search(r"== Working Agent: (.*?)\n", cleaned_data), re.search(r"== \[(.*?)\] Task output: (.*?)\n", cleaned_data, re.DOTALL)
        if agent_name_match and task_output_match:
            agent_name, task_output = agent_name_match.group(1), task_output_match.group(2)
            self.agent_task_outputs.append({"Agent": agent_name, "Output": task_output})
            st.toast(":robot_face: " + agent_name)
        for agent_name, color in self.agent_colors.items():
            cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")
        task_match = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE) or re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        if task_match:
            self.task_values.append(f"Task: {task_match.group(1).strip()}")
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []
