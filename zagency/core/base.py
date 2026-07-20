from typing import Callable

def tool(func: Callable) -> Callable:
    """
    Decorator to mark a method as a tool that can be called by an LM.
    """
    func._is_tool = True
    return func
