import time
import asyncio
from functools import wraps

def zimer(func=None, *, repeats=1):
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
    
def with_retry(num_retries=5, backoff=0, backoff_exponent=1):
    """
    Decorator to retry a function on exception.
    Args:
        num_retries (int): Number of retries. Default 5.
        backoff (float): Initial backoff in seconds. Default 0.
        backoff_exponent (float): Exponent for exponential backoff. Default 1 (linear).
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(num_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < num_retries - 1:
                        sleep_time = backoff * ((attempt + 1) ** backoff_exponent)
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    else:
                        raise last_exc
        return wrapper
    return decorator