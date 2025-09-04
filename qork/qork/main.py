import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

from qork.models import TokenCount
from qork.config import get_api_key, get_model
import time
from qork.utils import get_completion, get_token_count, get_response, QORK_ENV_KEY
from qork.session import get_session_key, load_previous_response_id, save_previous_response_id

def main():
    parser = argparse.ArgumentParser(
        description="A simple CLI for interacting with LLMs via litellm.",
        epilog="Example: qork \"What is the meaning of life?\" --model gpt-4o"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="The prompt to send to the model."
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help=f"The model to use for the completion. Defaults to QORK_MODEL env var or '{get_model()}'."
    )
    parser.add_argument(
        "-ns", "--no-stream",
        action="store_true",
        help="Disable streaming response from the model."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode. Overrides GPT_DEBUG_MODE environment variable."
    )
    parser.add_argument(
        "-r", "--responses",
        action="store_true",
        default=True,
        help="Use OpenAI Responses API backend (non-streaming)."
    )
    parser.add_argument(
        "-pt", "--plaintext",
        action="store_true",
        help="Print plain text output without rich formatting. Easier to copy/paste."
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use Chat Completions backend (overrides --responses)."
    )

    args = parser.parse_args()

    console = Console()

    if not args.prompt:
        parser.print_help()
        sys.exit(0)

    api_key = get_api_key()
    if not api_key:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable not set.[/bold red]")
        sys.exit(1)

    model_to_use = args.model if args.model else get_model()
    debug_mode = args.debug or os.environ.get("GPT_DEBUG_MODE", "false").lower() == "true"

    if args.chat:
        if not args.no_stream:
            stream_chat(model_to_use, args.prompt, api_key, console, debug_mode, args.plaintext)
        else:
            standard_chat(model_to_use, args.prompt, api_key, console, debug_mode, args.plaintext)
    else:
        model_to_use = args.model if args.model else "gpt-5-mini"
        responses_api_response(console, model_to_use, args.prompt, api_key, debug_mode, args.plaintext)

def stream_chat(model, prompt, api_key, console, debug_mode, plaintext=False):
    if debug_mode:
        if not plaintext:
            console.print("[yellow]Warning: Streaming token counts are estimated using the 'o200k_base' encoding and may not be exact.[/yellow]")
        else:
            print("WARNING: Streaming token counts are estimated using the 'o200k_base' encoding and may not be exact.")
        input_tokens = get_token_count(prompt)
        token_count = TokenCount(prompt_tokens=input_tokens)

    response = get_completion(model, prompt, api_key, stream=True)

    if plaintext:
        if isinstance(response, str) and response.startswith("Error:"):
            print(f"Error: {response}")
            if debug_mode:
                print(f"DEBUG: ErrorTokens: {token_count}")
            return
        for chunk in response:
            chunk_content: str = chunk.choices[0].delta.content
            if chunk_content:
                if debug_mode:
                    token_count.completion_tokens += get_token_count(chunk_content)
                print(chunk_content, end="", flush=True)
        print()
        if debug_mode:
            print(f"DEBUG: Tokens: {token_count}")
        return

    # rich flow
    with Live(Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"), console=console, screen=False, vertical_overflow="visible") as live:
        if isinstance(response, str) and response.startswith("Error:"):
            live.update(Panel(f"[bold red]{response}[/bold red]", title="Error", border_style="red"))
            if debug_mode:
                console.print(Text(f"ErrorTokens: {token_count}", style="dim"))
            return

        full_response = ""
        for chunk in response:
            chunk_content: str = chunk.choices[0].delta.content
            if chunk_content:
                full_response += chunk_content
                live.update(Panel(Markdown(full_response), title=f"[bold cyan]{model}[/bold cyan]", border_style="cyan", padding=(1, 2)))
                if debug_mode:
                    token_count.completion_tokens += get_token_count(chunk_content)
        if debug_mode:
            console.print(Text(f"Tokens: {token_count}", style="dim"))

def standard_chat(model, prompt, api_key, console, debug_mode, plaintext=False):
    response = get_completion(model, prompt, api_key)

    if plaintext:
        if isinstance(response, str) and response.startswith("Error:"):
            print(f"Error: {response}")
            return
        response_content = response.choices[0].message.content
        print(response_content)
        if debug_mode and response:
            usage = response.usage
            cost = response._hidden_params.get('response_cost', None)
            cost_str = f"${cost:.6f}" if isinstance(cost, (int, float)) else "N/A"
            debug_info = f"Model: {model} | Cost: {cost_str} | Total Tokens: {usage.total_tokens} [ Input: {usage.prompt_tokens} || Completion: {usage.completion_tokens} ]"
            print(f"DEBUG: {debug_info}")
        return

    # rich flow
    with Live(Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"), console=console, screen=False, vertical_overflow="visible") as live:
        if isinstance(response, str) and response.startswith("Error:"):
            live.update(Panel(f"[bold red]{response}[/bold red]", title="Error", border_style="red"))
            return

        response_content = response.choices[0].message.content
        output_panel = Panel(
            Markdown(response_content),
            title=f"[bold cyan]{model}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        live.update(output_panel)

    if debug_mode and response:
        usage = response.usage
        cost = response._hidden_params.get('response_cost', None)
        cost_str = f"${cost:.6f}" if isinstance(cost, (int, float)) else "N/A"
        debug_info = f"Model: {model} | Cost: {cost_str} | Total Tokens: {usage.total_tokens} [ Input: {usage.prompt_tokens} || Completion: {usage.completion_tokens} ]"
        console.print(Text(debug_info, style="dim"))

def responses_api_response(console, model, prompt, api_key, debug_mode, plaintext=False):
    try:
        # Per-shell session lookup: fetch a single previous_response_id
        session_key = get_session_key()
        previous_response_id = load_previous_response_id(session_key)

        response = get_response(
            console=console,
            model=model,
            prompt=prompt,
            api_key=api_key,
            stream=False,
            previous_response_id=previous_response_id,
        )
    except Exception as e:
        if plaintext:
            print(f"Error: {e}")
        else:
            console.print(Panel(f"[bold red]Error: {e}[/bold red]", title="Error", border_style="red"))
        return

    # Extract best-effort text content from Responses API object
    response_content = None
    try:
        response_content = getattr(response, "output_text", None)
    except Exception:
        response_content = None

    if not response_content:
        try:
            response_content = str(response)
        except Exception:
            response_content = "(No content)"

    if plaintext:
        print(response_content)
    else:
        output_panel = Panel(
            Markdown(response_content),
            title=f"[bold cyan]{model}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        console.print(output_panel)

    if debug_mode and response is not None:
        # Best-effort usage display; fields differ from Chat Completions
        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)
            debug_info = (
                f"Model: {model} | Total Tokens: {total_tokens if total_tokens is not None else 'N/A'}"
                f" [ Input: {prompt_tokens if prompt_tokens is not None else 'N/A'} || Output: {output_tokens if output_tokens is not None else 'N/A'} ]"
            )
        else:
            debug_info = f"Model: {model} | Usage: N/A"
        if plaintext:
            print(f"DEBUG: {debug_info}")
        else:
            console.print(Text(debug_info, style="dim"))

    # Persist the new previous_response_id after successful response
    try:
        session_key = get_session_key()
        conv_id = getattr(response, "id", None)
        if conv_id:
            save_previous_response_id(session_key, conv_id)
    except Exception:
        pass

if __name__ == "__main__":
    main()