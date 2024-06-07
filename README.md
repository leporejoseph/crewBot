![CrewBot](src/crew_ai/crewbot_assistant.png)

# CrewBot: Your AI Assistant
CrewBot is a Python-based AI application tailored to enhance business analysis and project management tasks. It leverages multiple AI capabilities, offering options between OpenAI's LLM, LM Studio, and Groq configurations. CrewBot is a full conversational chatbot that allows you to enable/disable crews in a given conversation, using the query and chat history as context. It uses pre-prompt engineering to assist task completion for the crews.

CrewBot assists users in the creation, editing, and running of a crew all within one chatbot interface, eliminating the need for coding. It allows the ability to run multiple crews in sequential order. CrewBot also features an export PDF option that generates a summary of your conversation history, providing a downloadable Streamlit button in the chat.

CrewBot will ask for required agent parameters if it does not have them and needs them to run the crew. You can switch the LLM at any given time and clear the chat history. Chat history, user preferences, and crews are all saved locally in the project as JSON files.

## Connect with Me

You can connect with me on LinkedIn:  [Joseph LePore](https://www.linkedin.com/in/joseph-lepore-062561b3)

---

## 🚀 Get Started 

### Quick Setup with Docker

To get started quickly, you can use Docker to set up the project.

```sh
# Build the Docker image
docker build -t crewbot .

# Run the Docker container
docker run -d -p 8501:8501 --name crewbot_app crewbot
```

### Quick Setup with Streamlit

If you prefer using Streamlit, follow these steps:

```sh
# Install the required libraries
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

### Environment Setup

Create a `.env` file in the root directory of your project if it does not already exists and add the following optional keys:

```plaintext
GROQ_API_KEY=Optional
OPENAI_API_KEY=Optional
BROWSERBASE_API_KEY=Optional
SERPER_API_KEY=Optional
BROWSERBASE_PROJECT_ID=Optional
```

## 💻 Technology Stack
- Python
- Streamlit
- LangChain
- CrewAi
- Docker

## 🤖 Available LLMs

### OpenAI
- **Pros**: High accuracy, wide range of capabilities.
- **Cons**: Requires API key, can be expensive.

### LM Studio
- **Pros**: Local hosting, customizable.
- **Cons**: Requires significant resources, complex setup.

### Groq
- **Pros**: High performance, scalable.
- **Cons**: Limited availability, requires API key.

## 📘 Tutorials

### Create a Crew
<details>
<summary>Step-by-step guide</summary>

1. Navigate to the "Create Crew" Section:
   - Go to the sidebar and click "Create a Crew."

2. Fill in the Crew Details:
   - Enter a name for your crew.
   - Select agents and tasks to include in the crew.

3. Configure Tools:
   - Select the necessary tools from the provided list.

4. Save the Crew:
   - Click "Create Crew" to save your new crew.

</details>

### Edit an Agent
<details>
<summary>Step-by-step guide</summary>

1. Select the Crew:
   - Navigate to the crew containing the agent you want to edit.

2. Edit Agent Details:
   - Click on the agent card to open the edit form.
   - Modify the agent's name, tools, goal, and backstory.

3. Save Changes:
   - Click "Save Agent" to apply the changes.

</details>

### Edit a Task
<details>
<summary>Step-by-step guide</summary>

1. Select the Crew:
   - Navigate to the crew containing the task you want to edit.

2. Edit Task Details:
   - Click on the task card to open the edit form.
   - Modify the task's description, tools, expected output, and context.

3. Save Changes:
   - Click "Save Task" to apply the changes.

</details>

### Run a Crew
<details>
<summary>Step-by-step guide</summary>

1. Select the Crew:
   - Navigate to the crew you want to run.

2. Run the Crew:
   - Click the "Run Crew" button to initiate the process.

3. Monitor the Progress:
   - Check the interaction logs for real-time updates.

</details>

### Langchain Tools
<details>
<summary>Step-by-step guide</summary>

1. Upload Documents:
   - Use the "Upload Documents" feature in the sidebar to add necessary files.

2. Summarize and Export:
   - Use the "Summarize and Export PDF" tool to generate a summary report.

</details>

## 🤓 Advanced Development

### Important File Functions List

<details>
<summary>src/app.py</summary>

- `initialize_app`: Sets up the initial application state.
- `init_session_state`: Initializes session state variables with defaults.
- `save_user_preferences`: Saves user preferences to a JSON file.
- `create_agents`: Creates agent instances for the crew.
- `create_tasks`: Creates task instances for the crew.
- `create_crew`: Creates and runs the crew process.
- `update_task_list`: Updates the task list in the UI.
- `update_agent_list`: Updates the agent list in the UI.
- `sidebar_configuration`: Configures the sidebar UI components.
- `create_new_agent_form`: Displays the form for creating a new agent.
- `create_new_task_form`: Displays the form for creating a new task.
- `create_new_crew_container`: Displays the container for creating a new crew.
- `display_crew_list`: Displays the list of existing crews in the sidebar.
- `edit_crew_agent`: Displays the form for editing an existing agent.
- `edit_crew_task`: Displays the form for editing an existing task.

</details>

<details>
<summary>src/crew_ai/crewai_utils.py</summary>

- `get_tool_instance`: Returns an instance of the specified tool.
- `create_agents`: Creates agent instances for the crew.
- `create_tasks`: Creates task instances for the crew.
- `create_crew`: Creates and runs the crew process.
- `update_crew_json`: Updates the specific crew in the crews.json file.
- `delete_crew`: Deletes a specific crew by index.

</details>

<details>
<summary>src/utils/llm_handler.py</summary>

- `init_llm`: Initializes the LLM based on the selected model.
- `set_initial_llm`: Sets the initial LLM state.
- `update_api_key`: Updates the API key for the selected LLM.
- `update_env_file`: Updates the environment file with a new key-value pair.
- `toggle_selection`: Toggles the selection between different LLMs.
- `get_response_async`: Asynchronously gets a response from the LLM.
- `get_response`: Synchronously gets a response from the LLM.

</details>


<details>
<summary>src/utils/document_handler.py</summary>

- `handle_document_upload`: Handles document upload and processing.
- `process_uploaded_files`: Processes uploaded files and returns a list of documents.
- `save_uploaded_file`: Saves the uploaded file to the specified path.
- `load_pdf_documents`: Loads PDF documents from the specified file path.
- `load_text_document`: Loads a text document from the uploaded file.
- `process_documents`: Processes documents for embedding and retrieval.
- `initialize_qa_chain`: Initializes the QA chain for document retrieval.
- `download_pdf`: Downloads a PDF of the summary report.

</details>

<details>
<summary>src/config.py</summary>

- `initialize_app`: Initializes the application and loads environment variables.
- `get_current_preferences`: Retrieves current user preferences.
- `save_user_preferences`: Saves user preferences to a JSON file.
- `preferences_changed`: Checks if user preferences have changed.
- `save_preferences_on_change`: Saves preferences when they change.
- `load_user_preferences`: Loads user preferences from a JSON file.
- `save_chat_history`: Saves chat history to a JSON file.
- `load_chat_history`: Loads chat history from a JSON file.
- `clear_chat_history`: Clears the chat history.
- `init_session_state`: Initializes session state variables with defaults.
- `get_card_styles`: Returns the styles for a card UI component.
- `get_empty_card_styles`: Returns the styles for an empty card UI component.

</details>

## 🕷️ Known Bugs
- Streamlit refresh and optimization issues with it.
- Editing an agent/task may cause UI issues.
- Inconsistencies with saving crew tools.

## 🙏 Future Improvements
- Custom tool creation for CrewAI built-in UI.
- Import/export crews in UI.
- Add more LLM settings like rate limiting, etc.

## 💸 Pricing, Accuracy, and Speed Examples Using Example Crew Business Requirements Crew

| LLM : Model            | Price per run | Accuracy | Speed  |
|----------------|---------------|----------|--------|
| OpenAI : Gpt4o         | $0.15 - $0.20           | Highest     | Fast   |
| LM Studio : dolphin-2.9-llama3-8b-GGUF       | Free            | Lowest: Depends on Hardware and Model   | Lowest: Depends on Hardware and Model |
| Groq : llama3-70b-8192           | Free tier but limited by tokens and requests per minute          | Medium     | Fastest   |

## License

This repository is released under the [Apache-2.0](LICENSE.md) License.
