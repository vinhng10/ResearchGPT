from collections import deque
import json
import os
from pathlib import Path
from typing import Any, Callable

from tenacity import stop_after_attempt, wait_exponential
from termcolor import colored


def build_tree(directory: str, ignore_list=None) -> str:
    PIPE = "│"
    ELBOW = "└──"
    TEE = "├──"
    PIPE_PREFIX = "│   "
    SPACE_PREFIX = "    "

    def _tree_head(directory: str) -> None:
        _tree.append(f"{directory}{os.sep}")

    def _tree_body(directory, prefix="") -> None:
        entries = _prepare_entries(directory)
        last_index = len(entries) - 1
        for index, entry in enumerate(entries):
            connector = ELBOW if index == last_index else TEE
            if entry.is_dir():
                if index == 0:
                    _tree.append(prefix + PIPE)
                _add_directory(entry, index, last_index, prefix, connector)
            else:
                _add_file(entry, prefix, connector)

    def _prepare_entries(directory):
        entries = filter(
            lambda entry: entry.name not in ignore_list, directory.iterdir()
        )
        entries = sorted(entries, key=lambda entry: str(entry))
        entries = sorted(entries, key=lambda entry: entry.is_file())
        return entries

    def _add_directory(directory, index, last_index, prefix, connector) -> None:
        _tree.append(f"{prefix}{connector} {directory.name}{os.sep}")
        if index != last_index:
            prefix += PIPE_PREFIX
        else:
            prefix += SPACE_PREFIX
        _tree_body(
            directory=directory,
            prefix=prefix,
        )
        if prefix := prefix.rstrip():
            _tree.append(prefix)

    def _add_file(file, prefix, connector) -> None:
        _tree.append(f"{prefix}{connector} {file.name}")

    directory = Path(directory)
    ignore_list = ignore_list if ignore_list else []
    _tree = deque()

    _tree_head(directory)
    _tree_body(directory)

    return "\n".join(_tree)


# Retry decorator settings
retry_settings = {
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "stop": stop_after_attempt(3),
}


# Pretty print decorator
def pretty_print(color: str) -> Callable[..., Any]:
    def decorator(func) -> Callable[..., Any]:
        def wrapper(*args, **kwargs) -> Any:
            output = func(*args, **kwargs)
            print_output = f"TOOL: {func.__name__}\nRESULT: {json.dumps(output)}\n"
            print(colored(print_output, color))
            return output

        return wrapper

    return decorator
