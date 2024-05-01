
# Crewbot: Your AI Assistant

Crewbot is a versatile Python-based AI application designed to assist with business analysis and project management. Leveraging OpenAI's LLM capabilities and crew-based task management, Crewbot provides comprehensive business reports by deploying specialized agents for market research, technological assessments, business model evaluations, and project management. The output is synthesized into a cohesive report for decision-making.

## Features

- **Market Research:** Provides a detailed analysis of market demand, customer profiles, and competitor strategies.
- **Technology Assessment:** Assesses technological feasibility, necessary tools, and architectural designs.
- **Business Evaluation:** Evaluates scalability, revenue streams, and sustainable business models.
- **Project Management:** Coordinates tasks, manages project scope, and tracks timelines for project development.
- **Comprehensive Reporting:** Summarizes all analyses into a cohesive Business Requirement Document, providing comprehensive project insights.

## Installation

1. **Python & Libraries:** Ensure Python and necessary libraries are installed by running:

   ```bash
   pip install streamlit langchain_core langchain_openai crewai langchain hub dotenv streamlit_chat pydantic
   ```

2. **Environment Setup:**
   - Edit the `.env` file in the project directory and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

   - For local LLM, install and configure Ollama and LM Studio:
      - [Download Ollama](https://ollama.com/).
      - [Download LM Studio](https://lmstudio.ai/) and choose an LLM model.

3. **Running Crewbot:**
   Start Crewbot by running:

   ```bash
   streamlit run app.py
   ```

## Usage:

1. **CrewAi: Run Business Analysis:** Select this checkbox and enter as much information about a business idea. Sit and watch as CrewAi agents build the Business Requirements for you.
2. **General Interaction:** Ask questions or give prompts in the chat input.

## Contribution Guidelines:

1. **Fork & Clone:** Fork this repository, then clone it to your machine.
2. **Branch:** Create a new branch for your feature or bug fix.
3. **PR:** Submit a Pull Request to the main branch.

## License:

[MIT License](LICENSE)

---

Crewbot is designed to streamline business analysis and project management, making decision-making easier and more informed.
