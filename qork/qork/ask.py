from typing import Optional

from rich.console import Console

from qork.config import get_api_key, get_model
from qork.models import TokenCount
from qork.utils import get_completion, get_response


def ask(
    prompt: str,
    model: Optional[str] = None,
    *,
    stream: Optional[bool] = None,
    responses: bool = False,
    plaintext: bool = True,
    debug: bool = False,
    return_text: bool = False,
):
    """
    Ask the model from Python (IPython/Jupyter) with the same behavior as the CLI.

    Parameters
    - prompt: The user prompt to send
    - model: Model name; defaults to env `QORK_MODEL` or library default
    - stream: Whether to stream tokens (default True, like CLI). Ignored for Responses API.
    - responses: Use OpenAI Responses API backend (non-streaming)
    - plaintext: Print plain text output (recommended for notebooks)
    - debug: Print token/cost info when available
    - return_text: If True, also return the response text

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
    effective_stream = True if stream is None else bool(stream)

    if responses:
        try:
            previous_response_id = None
            response = get_response(
                console=Console(),  # not used for output here
                model=chosen_model,
                prompt=prompt,
                api_key=api_key,
                previous_response_id=previous_response_id,
                stream=False,
            )
        except Exception as e:
            if plaintext:
                print(f"Error: {e}")
            else:
                Console().print(f"[bold red]Error: {e}[/bold red]")
            return None

        try:
            text = getattr(response, "output_text", None)
        except Exception:
            text = None
        if not text:
            try:
                text = str(response)
            except Exception:
                text = "(No content)"

        if plaintext:
            print(text)
        else:
            Console().print(text)

        if debug and response is not None:
            usage = getattr(response, "usage", None)
            if usage is not None:
                total_tokens = getattr(usage, "total_tokens", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                debug_info = (
                    f"Model: {chosen_model} | Total Tokens: {total_tokens if total_tokens is not None else 'N/A'}"
                    f" [ Input: {prompt_tokens if prompt_tokens is not None else 'N/A'} || Output: {output_tokens if output_tokens is not None else 'N/A'} ]"
                )
            else:
                debug_info = f"Model: {chosen_model} | Usage: N/A"
            if plaintext:
                print(f"DEBUG: {debug_info}")
            else:
                Console().print(f"[dim]{debug_info}[/dim]")

        return text if return_text else None

    # Chat Completions path
    if effective_stream:
        # estimate input tokens in debug mode
        token_count = None
        if debug:
            input_tokens = 0
            try:
                # lazy import to avoid overhead otherwise
                from qork.utils import get_token_count
                input_tokens = get_token_count(prompt)
            except Exception:
                input_tokens = 0
            token_count = TokenCount(prompt_tokens=input_tokens)

        response = get_completion(chosen_model, prompt, api_key, stream=True)
        if isinstance(response, str) and response.startswith("Error:"):
            if plaintext:
                print(response)
            else:
                Console().print(f"[bold red]{response}[/bold red]")
            return None

        collected = []
        for chunk in response:
            chunk_content = getattr(getattr(chunk.choices[0], "delta", object()), "content", None)
            if not chunk_content:
                continue
            collected.append(chunk_content)
            if plaintext:
                print(chunk_content, end="", flush=True)
            else:
                Console().print(chunk_content, end="")
            if debug and token_count is not None:
                try:
                    from qork.utils import get_token_count
                    token_count.completion_tokens += get_token_count(chunk_content)
                except Exception:
                    pass
        if plaintext:
            print()

        full_text = "".join(collected)
        if debug and token_count is not None:
            if plaintext:
                print(f"DEBUG: Tokens: {token_count}")
            else:
                Console().print(f"[dim]Tokens: {token_count}[/dim]")
        return full_text if return_text else None

    # Non-streaming completions
    response = get_completion(chosen_model, prompt, api_key)
    if isinstance(response, str) and response.startswith("Error:"):
        if plaintext:
            print(response)
        else:
            Console().print(f"[bold red]{response}[/bold red]")
        return None

    text = response.choices[0].message.content
    if plaintext:
        print(text)
    else:
        Console().print(text)

    if debug and response:
        try:
            usage = response.usage
            cost = response._hidden_params.get("response_cost", None)
            cost_str = f"${cost:.6f}" if isinstance(cost, (int, float)) else "N/A"
            debug_info = (
                f"Model: {chosen_model} | Cost: {cost_str} | Total Tokens: {usage.total_tokens}"
                f" [ Input: {usage.prompt_tokens} || Completion: {usage.completion_tokens} ]"
            )
            if plaintext:
                print(f"DEBUG: {debug_info}")
            else:
                Console().print(f"[dim]{debug_info}[/dim]")
        except Exception:
            pass

    return text if return_text else None


