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

        # Adjusted regex pattern to match "== Working Agent:" and capture the agent name
        agent_name_matches = re.findall(r"== Working Agent: (.*?)(?:\n|$)", cleaned_data)

        # Process all agent name matches
        for agent_name in agent_name_matches:
            st.toast(f"Starting Agent: {agent_name}")

            if agent_name not in self.agent_color_map:
                self.agent_color_map[agent_name] = self.agent_colors[self.current_color_index % len(self.agent_colors)]
                self.current_color_index += 1
            
            color = self.agent_color_map[agent_name]
            cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")

        self.buffer.append(cleaned_data)
        
        # Append data to buffer and check for newline to update the expander
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        pass
