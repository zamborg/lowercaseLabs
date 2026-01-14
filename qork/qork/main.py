from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from qork.config import get_api_key, get_model
from qork.session import (
    load_thread_response_id,
    save_thread_response_id,
)
from qork.utils import get_response, response_text, stream_response

app = typer.Typer(add_completion=False)

MODEL_PROFILES: dict[str, dict] = {
    "nano": {"model": "gpt-5-nano"},
    "mini": {"model": "gpt-5-mini"},
    "large": {"model": "gpt-5"},
    "high": {"model": "gpt-5", "reasoning": {"effort": "high"}},
}


def _print_debug(console: Console, *, model: str, response, plaintext: bool) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        msg = f"Model: {model} | Usage: N/A"
    else:
        total = getattr(usage, "total_tokens", None)
        input_tokens = getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None))
        output_tokens = getattr(usage, "output_tokens", None)
        msg = (
            f"Model: {model} | Total Tokens: {total if total is not None else 'N/A'}"
            f" [ Input: {input_tokens if input_tokens is not None else 'N/A'} || Output: {output_tokens if output_tokens is not None else 'N/A'} ]"
        )
    if plaintext:
        print(f"DEBUG: {msg}")
    else:
        console.print(Text(msg, style="dim"))


@app.callback(invoke_without_command=True)
def _cli(
    ctx: typer.Context,
    prompt: Optional[str] = typer.Argument(None, help="The prompt to send."),
    thread: bool = typer.Option(
        False,
        "-t",
        "--thread",
        help="Continue a single global thread stored at ~/.qork/history/session.id (clobbers across shells).",
    ),
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Model preset: nano|mini|large|high (high sets reasoning effort=high).",
    ),
    model: Optional[str] = typer.Option(None, "-m", "--model", help="Model name (default: QORK_MODEL or built-in default)."),
    stream: bool = typer.Option(False, "--stream/--no-stream", help="Stream output tokens."),
    plaintext: bool = typer.Option(False, "-pt", "--plaintext", help="Print plain text output (no Rich panels)."),
    debug: bool = typer.Option(False, "-d", "--debug", help="Print token usage when available."),
):
    console = Console()

    if not prompt:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)

    api_key = get_api_key()
    if not api_key:
        if plaintext:
            print("Error: OPENAI_API_KEY environment variable not set.")
        else:
            console.print("[bold red]Error: OPENAI_API_KEY environment variable not set.[/bold red]")
        raise typer.Exit(code=1)

    if model and profile:
        if plaintext:
            print("Error: Use either --model or --profile, not both.")
        else:
            console.print("[bold red]Error:[/bold red] Use either --model or --profile, not both.")
        raise typer.Exit(code=2)

    reasoning = None
    if profile:
        cfg = MODEL_PROFILES.get(profile)
        if not cfg:
            allowed = ", ".join(MODEL_PROFILES.keys())
            if plaintext:
                print(f"Error: Unknown profile '{profile}'. Allowed: {allowed}")
            else:
                console.print(f"[bold red]Error:[/bold red] Unknown profile '{profile}'. Allowed: {allowed}")
            raise typer.Exit(code=2)
        model_to_use = cfg["model"]
        reasoning = cfg.get("reasoning")
    else:
        model_to_use = model or get_model()

    previous_response_id = load_thread_response_id() if thread else None

    try:
        if stream:
            full_text = ""
            with stream_response(
                model=model_to_use,
                prompt=prompt,
                api_key=api_key,
                previous_response_id=previous_response_id,
                reasoning=reasoning,
            ) as s:
                if plaintext:
                    for event in s:
                        if getattr(event, "type", None) == "response.output_text.delta":
                            delta = getattr(event, "delta", "")
                            if delta:
                                full_text += delta
                                print(delta, end="", flush=True)
                    print()
                else:
                    with Live(
                        Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"),
                        console=console,
                        screen=False,
                        vertical_overflow="visible",
                    ) as live:
                        for event in s:
                            if getattr(event, "type", None) == "response.output_text.delta":
                                delta = getattr(event, "delta", "")
                                if not delta:
                                    continue
                                full_text += delta
                                live.update(
                                    Panel(
                                        Markdown(full_text),
                                        title=f"[bold cyan]{model_to_use}[/bold cyan]",
                                        border_style="cyan",
                                        padding=(1, 2),
                                    )
                                )

                response = s.get_final_response()
                text = response_text(response)
        else:
            response = get_response(
                model=model_to_use,
                prompt=prompt,
                api_key=api_key,
                previous_response_id=previous_response_id,
                reasoning=reasoning,
            )
            text = response_text(response)

            if plaintext:
                print(text)
            else:
                console.print(
                    Panel(
                        Markdown(text),
                        title=f"[bold cyan]{model_to_use}[/bold cyan]",
                        border_style="cyan",
                        padding=(1, 2),
                    )
                )
    except Exception as e:
        if plaintext:
            print(f"Error: {e}")
        else:
            console.print(Panel(f"[bold red]Error: {e}[/bold red]", title="Error", border_style="red"))
        raise typer.Exit(code=1)

    # Save session threading id (Responses API only)
    if thread:
        try:
            resp_id = getattr(response, "id", None)
            if isinstance(resp_id, str) and resp_id:
                save_thread_response_id(resp_id)
        except Exception:
            pass

    if debug:
        _print_debug(console, model=model_to_use, response=response, plaintext=plaintext)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
