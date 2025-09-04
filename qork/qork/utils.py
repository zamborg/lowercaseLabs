import os
import subprocess
from typing import Literal
from openai import OpenAI
from rich.console import Console
import litellm
import tiktoken

QORK_SYSTEM_PROMPT = "You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested."
QORK_ENV_KEY = "QORK_ENV_KEY"

BASH_TOOL_CALL = [{
    "type": "function",
    "name": "bash",
    "description": "Run a bash command",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to run"
            }
        },
        "required": ["command"]
    }
}]

ALLOWED_MODELS = Literal["gpt-5", "gpt-5-mini", "gpt-5-nano"]

def bash(command: str) -> str:
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout

def get_completion(model, prompt, api_key, stream=False):
    messages = [
        {
            "role": "system",
            "content": QORK_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = litellm.completion(
            model=model,
            messages=messages,
            api_key=api_key,
            stream=stream
        )
        return response
    except Exception as e:
        return f"Error: {e}"
    
def get_response(console: Console, model: ALLOWED_MODELS, prompt: str, api_key: str, previous_response_id: str = None, run_bash: bool = False, stream: bool = False) -> str:
    """
    Gets a response from the model
    - Presently does not support streaming
    """
    client = OpenAI(api_key=api_key)
    kwargs = {"model": model}
    if previous_response_id:
        # because there's a previous response id we do not prepend the system prompt
        messages = prompt
        kwargs["previous_response_id"] = previous_response_id
    else:
        messages = [
            {"role": "developer", "content": QORK_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

    if run_bash:
        kwargs["tools"] = BASH_TOOL_CALL # this is already a *list*
        kwargs["tool_choice"] = "required"
    
    kwargs["input"] = messages
    response = client.responses.create(**kwargs)

    results = []
    if run_bash: # this is broken for the time being
        for out in response.output:
            if out.type == "function_call" and out.name == "bash":
                results.append(bash(out.arguments["command"]))
        return "\n".join(results)
    else:
        return response # we just return the raw response
    
    

def get_token_count(content: str, encoding_name: str = "o200k_base") -> int:
    return len(tiktoken.get_encoding(encoding_name).encode(content))