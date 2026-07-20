#!/usr/bin/env python3
"""
Demo script showcasing QirkTheCoder - a state-of-the-art coding agent.
"""

import sys
from pathlib import Path

from core.base import LiteLLM
from agents.qirk_coder import QirkTheCoder
from environments.coding_environment import CodingEnvironment
from handler.step_handler import StepHandler


def demo_simple_coding_task():
    """Demo: Create a simple Python function with tests."""
    print("\n=== Demo: Simple Coding Task ===\n")
    
    # Create a temporary project directory
    project_dir = Path("./demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Initialize environment and agent
    env = CodingEnvironment(str(project_dir))
    lm = LiteLLM(model="gpt-4-turbo")
    qirk = QirkTheCoder(lm, str(project_dir), env)
    
    # Set up task
    env.update_state({
        "pending_tasks": [
            "Create a Python function called 'calculate_fibonacci' that returns the nth Fibonacci number",
            "Write comprehensive unit tests for the function",
            "Run the tests to ensure they pass"
        ]
    })
    
    # Run agent
    handler = StepHandler(env)
    handler.add_agent(qirk)
    handler.run(max_steps=5)


def demo_bug_fix():
    """Demo: Fix a bug in existing code."""
    print("\n=== Demo: Bug Fix ===\n")
    
    project_dir = Path("./demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create a buggy file
    buggy_code = '''def divide_numbers(a, b):
    """Divide two numbers."""
    return a / b  # Bug: No zero division check

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    total = sum(numbers)
    return divide_numbers(total, len(numbers))  # Bug: Empty list not handled
'''
    
    (project_dir / "math_utils.py").write_text(buggy_code)
    
    # Initialize environment and agent
    env = CodingEnvironment(str(project_dir))
    lm = LiteLLM(model="gpt-4-turbo")
    qirk = QirkTheCoder(lm, str(project_dir), env)
    
    # Set up bug fix task
    env.update_state({
        "pending_tasks": [
            "Read math_utils.py and identify potential bugs",
            "Fix the zero division error in divide_numbers function",
            "Fix the empty list handling in calculate_average function",
            "Add appropriate error handling and edge case tests"
        ]
    })
    
    # Run agent
    handler = StepHandler(env)
    handler.add_agent(qirk)
    handler.run(max_steps=6)


def demo_refactoring():
    """Demo: Refactor code for better structure."""
    print("\n=== Demo: Code Refactoring ===\n")
    
    project_dir = Path("./demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Create code that needs refactoring
    messy_code = '''# User management code
users = []

def add_user(name, email, age):
    # Add user to list
    user = {"name": name, "email": email, "age": age}
    users.append(user)
    print("User added: " + name)
    return True

def find_user(email):
    # Find user by email
    for u in users:
        if u["email"] == email:
            return u
    return None

def delete_user(email):
    # Delete user
    for i in range(len(users)):
        if users[i]["email"] == email:
            del users[i]
            return True
    return False

def list_users():
    # List all users
    for u in users:
        print(u["name"] + " - " + u["email"])
'''
    
    (project_dir / "user_manager.py").write_text(messy_code)
    
    # Initialize environment and agent
    env = CodingEnvironment(str(project_dir))
    lm = LiteLLM(model="gpt-4-turbo")
    qirk = QirkTheCoder(lm, str(project_dir), env)
    
    # Set up refactoring task
    env.update_state({
        "pending_tasks": [
            "Read user_manager.py and analyze the code structure",
            "Refactor the code into a proper UserManager class",
            "Add type hints and proper documentation",
            "Implement data validation for user inputs",
            "Create unit tests for the refactored code"
        ]
    })
    
    # Run agent
    handler = StepHandler(env)
    handler.add_agent(qirk)
    handler.run(max_steps=7)


def demo_multi_file_project():
    """Demo: Work with multiple files in a project."""
    print("\n=== Demo: Multi-File Project ===\n")
    
    project_dir = Path("./demo_project")
    project_dir.mkdir(exist_ok=True)
    
    # Initialize environment and agent
    env = CodingEnvironment(str(project_dir))
    lm = LiteLLM(model="gpt-4-turbo")
    qirk = QirkTheCoder(lm, str(project_dir), env)
    
    # Set up multi-file task
    env.update_state({
        "pending_tasks": [
            "Create a simple todo list application with the following structure: models.py for data models, storage.py for file-based storage, and cli.py for command-line interface",
            "Implement a Todo model with fields: id, title, description, completed, created_at",
            "Implement storage functions to save and load todos from a JSON file",
            "Create a CLI with commands: add, list, complete, delete",
            "Add error handling and input validation",
            "Create a README.md with usage instructions"
        ]
    })
    
    # Run agent
    handler = StepHandler(env)
    handler.add_agent(qirk)
    handler.run(max_steps=10)


def interactive_mode():
    """Interactive mode where user can give tasks to QirkTheCoder."""
    print("\n=== Interactive Mode ===\n")
    print("Enter your coding tasks (one per line, empty line to start):")
    
    tasks = []
    while True:
        task = input("> ")
        if not task:
            break
        tasks.append(task)
    
    if not tasks:
        print("No tasks provided. Exiting.")
        return
    
    # Use current directory as project
    project_dir = Path(".")
    
    # Initialize environment and agent
    env = CodingEnvironment(str(project_dir))
    lm = LiteLLM(model="gpt-4-turbo")
    qirk = QirkTheCoder(lm, str(project_dir), env)
    
    # Set up tasks
    env.update_state({"pending_tasks": tasks})
    
    # Run agent
    handler = StepHandler(env)
    handler.add_agent(qirk)
    handler.run(max_steps=len(tasks) * 2)  # Allow multiple steps per task


if __name__ == "__main__":
    demos = {
        "1": ("Simple Coding Task", demo_simple_coding_task),
        "2": ("Bug Fix", demo_bug_fix),
        "3": ("Code Refactoring", demo_refactoring),
        "4": ("Multi-File Project", demo_multi_file_project),
        "5": ("Interactive Mode", interactive_mode)
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, func = demos[sys.argv[1]]
        print(f"Running: {name}")
        func()
    else:
        print("QirkTheCoder Demo")
        print("=================")
        print("\nAvailable demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}: {name}")
        print("\nUsage: python demo_qirk_coder.py <demo_number>")
        print("\nExample: python demo_qirk_coder.py 1")