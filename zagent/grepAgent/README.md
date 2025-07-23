# GrepAgent - Function Finding Evaluation Framework

This directory contains an agent and environment implementation for evaluating function-finding capabilities using bash commands.

## Overview

GrepAgent is designed to test an agent's ability to:
1. Navigate file systems efficiently
2. Use bash commands (especially grep) to search for code
3. Identify function definitions across different programming languages
4. Submit accurate answers with minimal commands

## Architecture

### FileSystemEnvironment
- Manages a git repository (cloned or local)
- Executes bash commands safely with timeouts
- Tracks all commands and their outputs
- Records potential matches found by the agent

### GrepAgent
- Uses bash commands to search for function definitions
- Implements smart search strategies
- Parses grep output to identify matches
- Calculates confidence scores for matches
- Submits final answers when confident

### Evaluation Metrics
- **Accuracy**: Did the agent find the correct file?
- **Efficiency**: How many commands were needed?
- **Speed**: How long did it take?
- **Precision**: Did it find the exact line number?

## Usage

### Single Function Search
```bash
# Search in a remote repository
python grepAgent/run_grep_task.py \
    --function "my_function" \
    --repo "https://github.com/owner/repo.git" \
    --commit "abc123" \
    --expected "src/module.py"

# Search in a local repository
python grepAgent/run_grep_task.py \
    --function "my_function" \
    --local "/path/to/repo" \
    --expected "src/module.py"
```

### Dataset Evaluation
```bash
# Run on a dataset
python grepAgent/run_grep_task.py \
    --dataset grepAgent/example_dataset.json \
    --output results.json

# With options
python grepAgent/run_grep_task.py \
    --dataset dataset.json \
    --model gpt-4-turbo \
    --max-steps 30 \
    --output results.json
```

## Dataset Format

Datasets should be JSON files with the following structure:
```json
[
  {
    "function_name": "process_data",
    "git_repo": "https://github.com/owner/repo.git",
    "git_hash": "abc123def456",
    "filepath": "src/data/processor.py"
  }
]
```

## Implementation Details

### Search Strategies

The agent uses multiple strategies to find functions:

1. **Pattern-based search**: Generates language-specific patterns
   - Python: `def function_name(`
   - JavaScript: `function function_name(`, `const function_name =`
   - Go: `func function_name(`
   - etc.

2. **File filtering**: Uses `find` to locate relevant source files first

3. **Incremental refinement**: Starts broad, then narrows based on results

4. **Context verification**: Reads files to confirm function definitions

### Confidence Scoring

Matches are scored based on:
- Presence of definition keywords (def, function, func)
- File type (.py, .js, .go, etc.)
- Not being in test files (lower confidence)
- Exact word boundaries

### Command Execution

All commands are executed with:
- Timeout protection (default 30s)
- Output capture (stdout/stderr)
- Working directory set to repository root
- Command history tracking

## Example Commands

The agent might use commands like:
```bash
# Find all Python files
find . -name "*.py" -type f

# Search for function definition
grep -n "def my_function" --include="*.py" -r .

# Get context around match
grep -B2 -A5 "def my_function" src/module.py

# Check specific file
head -50 src/module.py
```

## Extending the Framework

### Adding Language Support

Edit `_generate_search_patterns()` in `grep_agent.py`:
```python
# Rust patterns
patterns.append(f"fn {function_name}\\(")
patterns.append(f"pub fn {function_name}\\(")
```

### Custom Evaluation Metrics

Extend the `evaluate()` method in `GrepTaskEnvironment`:
```python
# Add custom metrics
return {
    "success": exact_match,
    "metrics": {
        "grep_efficiency": grep_commands / total_commands,
        "search_breadth": len(unique_patterns_tried),
        # ... more metrics
    }
}
```

### Alternative Search Tools

The agent can use any bash command:
- `ag` (The Silver Searcher)
- `rg` (ripgrep) 
- `ack`
- `find` + `xargs`
- `ctags` integration

## Performance Considerations

1. **Repository Size**: Large repos may need adjusted timeouts
2. **Network Speed**: Cloning can be slow; consider caching
3. **Search Scope**: Limit search to relevant directories
4. **Pattern Complexity**: Balance precision vs. recall

## Future Enhancements

1. **Multi-language awareness**: Detect repo language and adjust patterns
2. **Semantic search**: Use AST parsing for better accuracy
3. **Learning**: Remember successful patterns for similar functions
4. **Parallel search**: Execute multiple searches simultaneously
5. **Caching**: Cache cloned repositories for repeated searches