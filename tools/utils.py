import json
from typing import Any, Callable

from tenacity import stop_after_attempt, wait_exponential
from termcolor import colored


# Retry decorator settings
retry_settings = {
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "stop": stop_after_attempt(3),
}


# Pretty print decorator
def pretty_print(color: str) -> Callable[..., Any]:
    def decorator(func):
        def wrapper(*args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            output = f"TOOL: {func.__name__}\nRESULT: {json.dumps(output)}\n"
            print(colored(output, color))
            return output
        return wrapper
    return decorator
