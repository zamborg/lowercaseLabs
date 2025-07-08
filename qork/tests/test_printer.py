import sys
import time

# Add the project root to the Python path to allow importing 'qork'
sys.path.insert(0, '/Users/zaysola/Documents/lowercaseLabs/qork')

from qork.printer import RichPrinter, ShellPrinter

def stream_generator():
    """A generator function to simulate streaming content."""
    text = "This is a demonstration of streaming output. Each word is yielded separately to simulate a real-time stream."
    for word in text.split():
        yield f"{word} "
        time.sleep(0.1)

def demonstrate_printers():
    """
    This script demonstrates the features of the RichPrinter and ShellPrinter classes.
    It is intended to showcase the visual output of each printer and serve as a
    basic integration test to ensure the printers are functioning correctly.
    """

    # --- RichPrinter Demonstration ---
    print("\n" + "="*50)
    print("--- Demonstrating RichPrinter ---")
    print("="*50 + "\n")

    rich_printer = RichPrinter(color="green", title="Rich Demo")

    print("\n>>> 1. RichPrinter - Static Output\n")
    rich_printer.print_static("This is a static message printed with RichPrinter. It supports Markdown!\n\n*   List item 1\n*   List item 2")

    print("\n>>> 2. RichPrinter - Streamed Output\n")
    rich_printer.print_stream(stream_generator())

    print("\n>>> 3. RichPrinter - Error Output\n")
    rich_printer.print_error("This is an error message from RichPrinter.")

    print("\n>>> 4. RichPrinter - Debug Output\n")
    rich_printer.print_debug("This is a debug message from RichPrinter.")


    # --- ShellPrinter Demonstration ---
    print("\n" + "="*50)
    print("--- Demonstrating ShellPrinter ---")
    print("="*50 + "\n")

    shell_printer = ShellPrinter()

    print("\n>>> 1. ShellPrinter - Static Output\n")
    shell_printer.print_static("This is a static message printed with ShellPrinter. It is plain text for easy copying.")

    print("\n>>> 2. ShellPrinter - Streamed Output\n")
    shell_printer.print_stream(stream_generator())

    print("\n>>> 3. ShellPrinter - Error Output\n")
    shell_printer.print_error("This is an error message from ShellPrinter.")

    print("\n>>> 4. ShellPrinter - Debug Output\n")
    shell_printer.print_debug("This is a debug message from ShellPrinter.")

    print("\n" + "="*50)
    print("--- Demonstration Complete ---")
    print("="*50 + "\n")


if __name__ == '__main__':
    demonstrate_printers()