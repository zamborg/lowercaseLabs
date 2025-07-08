from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from abc import ABC, abstractmethod

class BasePrinter(ABC):
    def __init__(self, color="cyan", title="Output"):
        self.color = color
        self.title = title

    @abstractmethod
    def print_stream(self, content_generator):
        pass

    @abstractmethod
    def print_static(self, content):
        pass

    @abstractmethod
    def print_error(self, content):
        pass

    @abstractmethod
    def print_debug(self, content):
        pass

class RichPrinter(BasePrinter):
    def __init__(self, color="cyan", title="Output"):
        super().__init__(color, title)
        self.console = Console()

    def print_stream(self, content_generator):
        with Live(Panel("[bold green]Querying...[/bold green]", title="Status", border_style="green"), console=self.console, screen=False, vertical_overflow="visible") as live:
            full_response = ""
            for chunk in content_generator:
                full_response += chunk
                live.update(Panel(Markdown(full_response), title=f"[bold {self.color}]{self.title}[/bold {self.color}]", border_style=self.color, padding=(1, 2)))

    def print_static(self, content):
        output_panel = Panel(
            Markdown(content),
            title=f"[bold {self.color}]{self.title}[/bold {self.color}]",
            border_style=self.color,
            padding=(1, 2)
        )
        self.console.print(output_panel)

    def print_error(self, content):
        self.console.print(Panel(f"[bold red]{content}[/bold red]", title="Error", border_style="red"))

    def print_debug(self, content):
        self.console.print(Text(content, style="dim"))

class ShellPrinter(BasePrinter):
    def print_stream(self, content_generator):
        for chunk in content_generator:
            print(chunk, end="", flush=True)
        print()

    def print_static(self, content):
        print(content)

    def print_error(self, content):
        print(f"Error: {content}")

    def print_debug(self, content):
        print(f"DEBUG: {content}")
