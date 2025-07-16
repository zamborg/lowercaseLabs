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
    
def with_retry(func=None, *, num_retries=5, backoff=0, backoff_exponent=1):
    """
    Decorator to retry a function on exception.

    Can be used with or without arguments:

    @with_retry
    def my_func():
        ...

    @with_retry(num_retries=3, backoff=1)
    def another_func():
        ...

    Args:
        num_retries (int): Number of retries. Must be positive.
        backoff (float): Initial backoff in seconds. Must be >= 0.
        backoff_exponent (float): Exponent for exponential backoff. Must be >= 0.
    """
    # When the decorator is invoked with arguments (i.e. @with_retry(...))
    if func is None:
        def decorator(fn):
            return with_retry(
                fn,
                num_retries=num_retries,
                backoff=backoff,
                backoff_exponent=backoff_exponent,
            )
        return decorator

    # Basic validation of parameters
    if not isinstance(num_retries, int) or num_retries < 1:
        raise ValueError("num_retries must be a positive integer")
    if backoff < 0:
        raise ValueError("backoff must be non-negative")
    if backoff_exponent < 0:
        raise ValueError("backoff_exponent must be non-negative")

    # Async variant
    if asyncio.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(num_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < num_retries - 1:
                        sleep_time = backoff * ((attempt + 1) ** backoff_exponent)
                        if sleep_time > 0:
                            await asyncio.sleep(sleep_time)
            # Exhausted retries – re-raise the last exception
            raise last_exc
        return async_wrapper

    # Sync variant
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
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
        # Exhausted retries – re-raise the last exception
        raise last_exc

    return sync_wrapper