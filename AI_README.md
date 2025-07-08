# AI Generated README

This README is an AI-generated overview of the `lowercaseLabs` repository. It is intended to be a comprehensive and detailed guide to the projects, code, and data within this repository. This document is updated regularly to reflect the current state of the repository.

## Repository Overview

The `lowercaseLabs` repository is a collection of various projects, primarily focused on Python-based data science, machine learning, and command-line tools. The repository is organized into several subdirectories, each containing a distinct project or a set of related files.

### Top-Level Directory Structure

```
/Users/zaysola/Documents/lowercaseLabs/
├───.gitignore
├───README.md
├───.git/...
├───ClothesPicker/
├───dump/
├───gpt_shell/
├───jupyterDocker/
├───qork/
├───raggar/
├───research/
├───sapProject/
└───timeit/
```

## Project Descriptions

### ClothesPicker

This project is a tool for planning outfits, leveraging image captioning and a vector database for clothing retrieval.

-   **`utils.py`**: Contains the core logic, including:
    -   `Preprocess`: A class with static methods for image manipulation (resize, background removal, greyscale, square crop).
    -   `Clothing`: A class to represent a single clothing item, with properties for its attributes (color, category, etc.) and a preprocessed image.
    -   `AllClothes`: A container class for all clothing items, with methods to load from a database.
    -   `ImageCaptioner`: Uses the `bardapi` to generate descriptions for clothing images.
    -   `VectorDB`: A custom vector database implementation using `sentence-transformers/all-MiniLM-L6-v2` to embed clothing descriptions and find nearest neighbors.
-   **`Test.ipynb`**: A Jupyter Notebook for testing the functionalities in `utils.py`, including image preprocessing, captioning, and vector database operations.
-   **Database Interaction**: The project uses `psycopg2` to connect to a PostgreSQL database to store and retrieve clothing information.
-   **Prompts**: `CLOTHES_INSTRUCTION.prompt` and `CLOTHES_LIST.prompt` are used to guide the outfit planning agent.

### dump

This directory contains `Classification.ipynb`, a Jupyter Notebook that experiments with a simple neural network for text classification. It uses `torch` for model building and `wandb` for logging. The notebook loads data from the `allenai/real-toxicity-prompts` and `google/civil_comments` datasets.

### gpt_shell

This project provides `gpt.sh`, a powerful shell script for interacting with OpenAI's GPT models from the command line.

-   **Features**:
    -   Supports different models via command-line arguments.
    -   Includes helper functions `nano` (for `gpt-4.1-nano`) and `mini` (for `gpt-4.1-mini`).
    -   Provides a debug mode to display token usage and cost, controlled by the `GPT_DEBUG_MODE` environment variable.
-   **Dependencies**: Requires `jq` for JSON parsing.
-   **Configuration**: Requires the `OPENAI_API_KEY` environment variable to be set.

### jupyterDocker

This project contains files for setting up a Dockerized Jupyter Notebook environment.

-   **`Dockerfile`**: Builds a Docker image based on an AWS PyTorch training image (`pytorch-training:2.1.0-cpu-py310-ubuntu20.04-ec2`). It installs dependencies from `requirements.txt` and sets up the Jupyter environment.
-   **Scripts**:
    -   `setup.sh`: Authenticates with AWS ECR and pulls the base Docker images.
    -   `run.sh`: Starts the Jupyter Lab server within the container.
    -   `docker_push.sh`: Builds and pushes the Docker image to an AWS ECR repository.

### qork

`qork` is a simple, beautiful, and powerful CLI for interacting with LLMs.

-   **Architecture**:
    -   `main.py`: The entry point, using `argparse` to handle command-line arguments.
    -   `utils.py`: Contains `get_completion`, which uses `litellm` to send prompts to various LLMs.
    -   `config.py`: Manages configuration from environment variables (`OPENAI_API_KEY`, `QORK_MODEL`).
    -   `printer.py`: Uses the `rich` library to provide formatted and visually appealing output in the terminal.
    -   `models.py`: Defines a `TokenCount` Pydantic model for tracking token usage.
-   **`AGENTS.md`**: Provides detailed documentation on the `qork` agent's architecture and functionality.

### raggar

`RAGGAR` is a Python project implementing a Retrieval Augmented Generation (RAG) and Generation Augmented Retrieval (GAR) framework.

-   **`chatter/`**: An example implementation of the RAGGAR framework.
    -   `agents.py`: Defines agents for searching (`SearchAgent` using `duckduckgo_search`) and interacting with OpenAI models (`OpenAIAgent`).
    -   `gen.py`: Contains the `ChatAgent`, which orchestrates the search and generation process to answer user queries.

### research

This directory contains research-related documents.

-   **`SAP/ResearchSummary.md`**: Contains summaries of several research papers related to LLM alignment, bias, and evaluation.
-   **`unfiltered/2109.03646.pdf`**: A PDF of a research paper.

### sapProject

This project is related to a study on African American Language (AAL).

-   **Data**:
    -   `AAVE.jsonl`, `WAE.jsonl`: JSONL files containing text data with toxicity scores.
    -   `AAL dataset/`: Contains the dataset from Deas et al. (2023), "Evaluation of African American Language Bias in Natural Language Generation," with paired AAL and White Mainstream English (WME) texts.
-   **`PrelimAnalysis.ipynb`**: A Jupyter Notebook for performing preliminary analysis on the AAVE and WAE datasets, including statistical comparisons and visualizations.

### timeit

`ztime` is a simple Python decorator for timing function execution.

-   **`ztime/__init__.py`**: Contains the implementation of the `ztime` decorator, which can be used with or without a `repeats` argument to time a function's execution and print the average time.
-   **`tests/test_ztime.py`**: Includes unit tests for the `ztime` decorator.

## Other Files

-   `.gitignore`: A file specifying which files and directories to ignore in Git.
-   `README.md`: The main README for the repository.

This AI-generated README will be updated as the repository evolves.
