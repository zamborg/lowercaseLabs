import time
import asyncio
from functools import wraps

def ztime(func=None, *, repeats=1):
    """
    A decorator to time the execution of a function.

    Can be used with or without arguments:
    @ztime
    def my_func():
        ...

    @ztime(repeats=3)
    def another_func():
        ...
    """
    if not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")

    def decorator(fn):
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                total_time = 0
                result = None
                for _ in range(repeats):
                    start_time = time.perf_counter()
                    result = await fn(*args, **kwargs)
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                avg_time = total_time / repeats
                print(f"Function '{fn.__name__}' took an average of {avg_time:.4f} seconds over {repeats} run(s).")
                return result
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                total_time = 0
                result = None
                for _ in range(repeats):
                    start_time = time.perf_counter()
                    result = fn(*args, **kwargs)
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                avg_time = total_time / repeats
                print(f"Function '{fn.__name__}' took an average of {avg_time:.4f} seconds over {repeats} run(s).")
                return result
            return sync_wrapper

    if func is None:
        # Called with arguments, e.g., @ztime(repeats=3)
        return decorator
    else:
        # Called without arguments, e.g., @ztime
        return decorator(func)