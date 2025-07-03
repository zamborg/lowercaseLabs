import os

def get_api_key():
    return os.environ.get("OPENAI_API_KEY")

def get_model():
    return os.environ.get("QORK_MODEL", "gpt-4.1-mini")
