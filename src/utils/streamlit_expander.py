# src/utils/streamlit_expander.py
import streamlit as st
import re

class StreamToExpander:
    """Redirect output to a Streamlit expander with formatted text and color coding for different agents."""
    def __init__(self, expander, agent_task_outputs):
        self.expander = expander
        self.agent_task_outputs = agent_task_outputs
        self.buffer = []
        self.agent_colors = ["blue", "red", "green", "violet", "orange"]
        self.current_color_index = 0
        self.agent_color_map = {}

    def write(self, data):
        """Process and format the input data, extract agent names and task outputs, and append to buffer for display."""
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)
        agent_name_match = re.search(r"== Working Agent: (.*?)\n", cleaned_data)
        task_output_match = re.search(r"== \[(.*?)\] Task output: (.*?)\n", cleaned_data, re.DOTALL)
        
        if agent_name_match and task_output_match:
            agent_name = agent_name_match.group(1)
            task_output = task_output_match.group(2)
            self.agent_task_outputs.append({"Agent": agent_name, "Output": task_output})
            st.toast(":robot_face: " + agent_name)
        
        if agent_name_match:
            agent_name = agent_name_match.group(1)
            if agent_name not in self.agent_color_map:
                self.agent_color_map[agent_name] = self.agent_colors[self.current_color_index % len(self.agent_colors)]
                self.current_color_index += 1
            color = self.agent_color_map[agent_name]
            cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")
        
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []