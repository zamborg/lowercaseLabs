from typing import Optional

from rich.console import Console

from qork.config import get_api_key, get_model
from qork.session import load_thread_response_id, save_thread_response_id
from qork.utils import get_response, response_text, stream_response


def ask(
    prompt: str,
    model: Optional[str] = None,
    *,
    stream: bool = False,
    profile: Optional[str] = None,
    thread: bool = False,
    plaintext: bool = True,
    debug: bool = False,
    previous_response_id: Optional[str] = None,
    return_text: bool = False,
):
    """
    Ask the model from Python (IPython/Jupyter).

    Parameters
    - prompt: The user prompt to send
    - model: Model name; defaults to env `QORK_MODEL` or built-in default
    - stream: Stream output tokens (Responses API)
    - profile: Model preset: nano|mini|large|high (high sets reasoning effort=high)
    - thread: Continue a single global thread stored at ~/.qork/history/session.id (clobbers across shells)
    - plaintext: Print plain text output (recommended for notebooks/scripts)
    - debug: Print token usage when available
    - previous_response_id: Continue a thread via Responses API
    - return_text: If True, also return the response text value

    Returns
    - str | None: The text output if return_text=True, else None
    """
    api_key = get_api_key()
    if not api_key:
        msg = "Error: OPENAI_API_KEY environment variable not set."
        if plaintext:
            print(msg)
        else:
            Console().print(f"[bold red]{msg}[/bold red]")
        return None

    chosen_model = model or get_model()
    reasoning = None
    if profile:
        profiles = {
            "nano": {"model": "gpt-5-nano"},
            "mini": {"model": "gpt-5-mini"},
            "large": {"model": "gpt-5"},
            "high": {"model": "gpt-5", "reasoning": {"effort": "high"}},
        }
        cfg = profiles.get(profile)
        if not cfg:
            allowed = ", ".join(profiles.keys())
            print(f"Error: Unknown profile '{profile}'. Allowed: {allowed}")
            return None
        chosen_model = cfg["model"]
        reasoning = cfg.get("reasoning")
    if model and profile:
        print("Error: Use either model=... or profile=..., not both.")
        return None

    effective_previous_response_id = previous_response_id
    if effective_previous_response_id is None and thread:
        effective_previous_response_id = load_thread_response_id()

    try:
        if stream:
            collected: list[str] = []
            with stream_response(
                model=chosen_model,
                prompt=prompt,
                api_key=api_key,
                previous_response_id=effective_previous_response_id,
                reasoning=reasoning,
            ) as s:
                for event in s:
                    if getattr(event, "type", None) == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if not delta:
                            continue
                        collected.append(delta)
                        if plaintext:
                            print(delta, end="", flush=True)
                        else:
                            Console().print(delta, end="")
                if plaintext:
                    print()
                response = s.get_final_response()
                text = response_text(response)
        else:
            response = get_response(
                model=chosen_model,
                prompt=prompt,
                api_key=api_key,
                previous_response_id=effective_previous_response_id,
                reasoning=reasoning,
            )
            text = response_text(response)
            if plaintext:
                print(text)
            else:
                Console().print(text)
    except Exception as e:
        if plaintext:
            print(f"Error: {e}")
        else:
            Console().print(f"[bold red]Error: {e}[/bold red]")
        return None

    if thread:
        try:
            resp_id = getattr(response, "id", None)
            if isinstance(resp_id, str) and resp_id:
                save_thread_response_id(resp_id)
        except Exception:
            pass

    if debug and response is not None:
        usage = getattr(response, "usage", None)
        if usage is None:
            debug_info = f"Model: {chosen_model} | Usage: N/A"
        else:
            total_tokens = getattr(usage, "total_tokens", None)
            input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
            output_tokens = getattr(usage, "output_tokens", None)
            debug_info = (
                f"Model: {chosen_model} | Total Tokens: {total_tokens if total_tokens is not None else 'N/A'}"
                f" [ Input: {input_tokens if input_tokens is not None else 'N/A'} || Output: {output_tokens if output_tokens is not None else 'N/A'} ]"
            )
        if plaintext:
            print(f"DEBUG: {debug_info}")
        else:
            Console().print(f"[dim]{debug_info}[/dim]")

    return text if return_text else None
