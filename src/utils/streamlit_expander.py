# src/utils/streamlit_expander.py
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import re

class StreamToExpander:
    """Redirect output to a Streamlit expander with formatted text and color coding for different agents."""
    def __init__(self, expander, crew_name, buffer_limit=10000):
        self.expander = expander
        self.crew_name = crew_name
        self.buffer = []
        self.crew_results = ""
        self.results_finished = False
        self.buffer_limit = buffer_limit
        self.agent_task_outputs = {}
        self.agent_colors = ["blue", "red", "green", "violet", "orange"]
        self.current_color_index = 0
        self.agent_color_map = {}
        self.chat_messages_history = StreamlitChatMessageHistory(key='chat_messages')

    def write(self, data):
        """Process and format the input data, extract agent names and task outputs, and append to buffer for display."""
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Adjusted regex pattern to match "== Working Agent:" and capture the agent name
        agent_name_matches = re.findall(r"== Working Agent: (.*?)(?:\n|$)", cleaned_data)

        # Process all agent name matches
        for agent_name in agent_name_matches:
            st.toast(":robot_face: Starting Agent: " + f"{agent_name}")

            if agent_name not in self.agent_color_map:
                self.agent_color_map[agent_name] = self.agent_colors[self.current_color_index % len(self.agent_colors)]
                self.current_color_index += 1

            color = self.agent_color_map[agent_name]
            cleaned_data = cleaned_data.replace(agent_name, f":{color}[{agent_name}]")

        self.buffer.append(cleaned_data)
        self.crew_results += cleaned_data

        # Check for the final answer using a more flexible regex pattern
        final_answer_matches = re.findall(r"final\s*answer:\s*(.*?)(?:\n|$)", self.crew_results, re.IGNORECASE)
        if final_answer_matches and not self.results_finished:
            self.chat_messages_history.add_ai_message(self.crew_results)
            self.results_finished = True

        if "\n" in data:
            full_message = ''.join(self.buffer)
            self.expander.markdown(full_message, unsafe_allow_html=True)
            self.buffer.clear()

    def flush(self):
        pass
