import litellm

def get_completion(model, prompt, api_key, stream=False):
    messages = [
        {
            "role": "system",
            "content": "You are a commandline assistant. The user is a sophisticated developer looking for a FAST and ACCURATE answer to their question. You should be concise and to the point. Prioritize answers, and explanations ONLY when requested."
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
