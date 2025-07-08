# qork

A simple, beautiful, and powerful CLI for interacting with LLMs.

## Installation

```bash
pip install qork
```

## Usage

```bash
qork "Your prompt here"
```

### Streaming

By default, `qork` streams the response from the model for a more interactive experience. You can disable this with the `--no-stream` flag:

### Debug Mode

To see token usage and cost information, use the `-d` or `--debug` flag:

```bash
qork "What is the capital of Japan?" -d
```

This will display a panel with the following information:

```
Model: gpt-4.1-mini | Cost: $0.000012 | Total Tokens: 23 [ Input: 15 || Completion: 8 ]
```

### Selecting a Model

You can specify a model using the `-m` or `--model` flag:

```bash
qork "Translate 'hello' to French." --model gpt-4o
```

You can also set the `QORK_MODEL` environment variable to change the default model.