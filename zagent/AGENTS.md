
# Available Agents

This document provides an overview of the agents available in this framework.

## 1. `PreferenceMapper`

The `PreferenceMapper` agent is designed to interact with a user to determine their preferences for a specific task.

### Functionality

- **Interviews the user:** Asks a series of questions to understand the user's needs and requirements.
- **Saves preferences:** Once the interview is complete, it synthesizes the user's preferences into a detailed summary and saves it to a markdown file.

### Tools

- `ask_user(question: str)`: Asks the user a question.
- `save_preferences(preferences: str, filename: str)`: Saves the final preferences to a file.
- `exit()`: Stops the agent's execution.

### Usage

This agent is useful for tasks that require gathering user input before execution, such as configuring a report or personalizing an analysis.

## 2. `CodeEditor`

The `CodeEditor` agent is a simple agent that can view and modify the content of a single file.

### Functionality

- **Views a file:** Can read and display the content of a file.
- **Updates a file:** Can modify the content of the file.

### Tools

- `view_file()`: Returns the current content of the file.
- `update_file(new_content: str)`: Replaces the file's content with new content.

### Usage

This agent is a basic building block for more complex agents that need to interact with the file system.

## 3. `QirkTheCoder`

`QirkTheCoder` is a more advanced version of the `CodeEditor` agent. It can not only modify files but also execute shell commands.

### Functionality

- **File modification:** Can write content to a specified file path.
- **Command execution:** Can run arbitrary bash commands and return the output.

### Tools

- `update_file(file_path: str, new_content: str)`: Writes new content to a file.
- `run_bash_command(command: str)`: Executes a bash command.

### Configuration

This agent's behavior is configured through a `qirkCode.yaml` file, which defines the prompts used by the agent.

### Usage

`QirkTheCoder` is a powerful agent that can be used for a wide range of software development tasks, such as:

- Writing and modifying code.
- Running tests and builds.
- Interacting with version control systems.
