# CrewBot: Your AI Assistant

CrewBot is a Python-based AI application tailored to enhance business analysis and project management tasks. It leverages multiple AI capabilities, offering options between OpenAI's LLM and local LLM configurations through LM Studio. CrewBot efficiently manages tasks and produces detailed business reports utilizing crew-based task execution.

## Features

- **Dynamic LLM Selection**: Users can toggle between OpenAI's GPT models and local LLMs for flexibility in AI operations.
- **Enhanced User Interaction**: Provides a rich interface for users to interact, ask questions, or provide input via an intuitive chat interface.
- **Crew-based Task Execution**: Integrates the capability to execute business analysis tasks by deploying specialized agents for various analyses.
- **Customizable AI Environment**: Allows users to configure AI settings such as API keys and local server URLs directly through the interface.
- **Secure API Key Management**: Facilitates secure handling and updating of API keys via the application's interface.

## Technology Stack

- **Streamlit**: For creating and managing the web app interface.
- **CrewAi**: For implementing task-based AI agents that handle business analysis and reporting.
- **Langchain**: To facilitate interaction with large language models and enable complex AI-driven operations.

## Installation

Ensure Python and necessary libraries are installed by executing the following command:

```bash
pip install streamlit langchain_core langchain_openai crewai langchain dotenv streamlit_chat pydantic
```

### Environment Setup

1. **API Key Configuration**:
    - Modify the `.env` file in the project directory to include your OpenAI API key:
      ```plaintext
      OPENAI_API_KEY=your_openai_api_key
      ```

2. **Local LLM Setup**:
    - Download and set up Ollama and LM Studio for local LLM usage.
    - Select and configure an LLM model in LM Studio.

### Running CrewBot

Start the application by running:

```bash
streamlit run app.py
```

## Usage

- **LLM Selection**: Choose between using OpenAI's LLM or a local LLM model via the sidebar.
- **Input Handling**: Input your queries or information in the chat interface to receive AI-generated responses or business analyses.
- **Business Analysis**: When enabled, CrewBot uses a crew of AI agents to generate detailed business requirement documents based on the input provided.

## Contribution Guidelines

1. **Fork & Clone**: Fork this repository, then clone it to your machine.
2. **Create a Branch**: Make a new branch for your proposed feature or bug fix.
3. **Submit a Pull Request**: Push your branch and changes, then submit a pull request to the main branch.

## License

This project is licensed under the MIT License, supporting open collaboration.

CrewBot is designed to make business analysis and project management straightforward, enhancing decision-making through AI-driven insights.
