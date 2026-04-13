# Task: Add a greeting function

## Problem
The `main.py` file has a basic print statement but no reusable greeting function.

## Requirements
- Add a `greet(name: str) -> str` function that returns `"Hello, {name}!"`
- Update the `if __name__ == "__main__"` block to use the function
- Add a test file `test_main.py` that tests the greeting function with at least 3 cases

## Acceptance Criteria
- `python -m pytest test_main.py -v` passes
- The function handles empty string and None gracefully
