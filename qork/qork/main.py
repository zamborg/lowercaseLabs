import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from .config import get_api_key, get_model
from .utils import get_completion

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
        "-s", "--stream",
        action="store_true",
        help="Stream the response from the model."
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode. Overrides GPT_DEBUG_MODE environment variable."
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

    if args.stream and debug_mode:
        console.print("[bold yellow]Warning: Debug mode is not supported with streaming. The -d flag will be ignored.[/bold yellow]")

    if args.stream:
        stream_response(model_to_use, args.prompt, api_key, console)
    else:
        standard_response(model_to_use, args.prompt, api_key, console, debug_mode)

def stream_response(model, prompt, api_key, console):
    with Live(Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"), console=console, screen=False, vertical_overflow="visible") as live:
        response = get_completion(model, prompt, api_key, stream=True)
        
        if isinstance(response, str) and response.startswith("Error:"):
            live.update(Panel(f"[bold red]{response}[/bold red]", title="Error", border_style="red"))
            return

        full_response = ""
        for chunk in response:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content:
                full_response += chunk_content
                live.update(Panel(Markdown(full_response), title=f"[bold cyan]{model}[/bold cyan]", border_style="cyan", padding=(1, 2)))

def standard_response(model, prompt, api_key, console, debug_mode):
    response = None
    with Live(Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"), console=console, screen=False, vertical_overflow="visible") as live:
        response = get_completion(model, prompt, api_key)

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

if __name__ == "__main__":
    main()