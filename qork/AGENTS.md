# qork Agent Documentation

This document provides an overview of the `qork` agent, a command-line interface (CLI) tool for interacting with Large Language Models (LLMs).

## Overview

`qork` is a Python-based CLI tool that allows users to send prompts to various LLMs and receive responses directly in their terminal. It is designed to be simple, efficient, and developer-focused. The tool supports features like streaming responses, selecting different models, and displaying debugging information such as token usage and cost.

## How it Works

The agent is built using the following key components:

-   **`litellm`**: This library provides a unified interface to interact with a wide range of LLMs, including those from OpenAI, Anthropic, and others. `qork` uses `litellm` to handle all communication with the selected language model.
-   **`rich`**: For providing a visually appealing and readable output in the terminal, `qork` uses the `rich` library. It is used for rendering Markdown, creating panels, and displaying status messages.
-   **`argparse`**: The command-line arguments (`--prompt`, `--model`, `--stream`, `--debug`) are parsed using Python's built-in `argparse` library.

The core logic is contained within the `qork` package:

-   **`main.py`**: This is the entry point of the application. It handles parsing command-line arguments, retrieving the API key and model configuration, and orchestrating the call to the LLM. It includes two main functions for handling responses: `stream_response` for interactive streaming and `standard_response` for a single, complete output.
-   **`config.py`**: This module is responsible for managing configuration, such as retrieving the `OPENAI_API_KEY` and the desired `QORK_MODEL` from environment variables.
-   **`utils.py`**: This file contains the `get_completion` function, which constructs the message payload (including the system prompt) and calls the `litellm.completion` method to get the response from the model.

## System Prompt

To ensure concise and accurate answers suitable for a developer audience, the following system prompt is sent with every user request:

> You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested.

## Dependencies

The agent relies on the following Python libraries:

-   `litellm`: For interacting with LLMs.
-   `rich`: For rich text and beautiful formatting in the terminal.

These are listed in the `pyproject.toml` file.

## Configuration

The agent can be configured using environment variables:

-   **`OPENAI_API_KEY`**: Your API key for the LLM service (e.g., OpenAI). This is required.
-   **`QORK_MODEL`**: The default model to use for completions. This can be overridden with the `--model` flag. If not set, it defaults to `gpt-4.1-mini`.

## Usage

To use the agent, you can run it from the command line with a prompt:

```bash
qork "What is the capital of France?"
```

For more advanced usage, including streaming and model selection, refer to the `README.md` file.
