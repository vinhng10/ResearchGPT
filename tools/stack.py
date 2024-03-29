from typing import List
from tenacity import retry
from .utils import pretty_print, retry_settings


PUSH_TASK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "push_task",
        "description": "Push a task to a task stack.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task to pushed.",
                },
            },
            "required": ["task"],
        },
    },
}

POP_TASK_SCHEMA = {
    "type": "function",
    "function": {
        "name": "pop_task",
        "description": "Pop a task from a task stack.",
    },
}


@retry(**retry_settings)
@pretty_print(color="yellow")
def push_task(task: str, stack: List[str]) -> None:
    """
    Push a task to a task stack.

    Args:
        task (str): The task.
        stack (List[str]): The task stack.

    Returns:
        None
    """
    try:
        stack.append(task)
        return task
    except Exception as e:
        return e


@retry(**retry_settings)
@pretty_print(color="yellow")
def pop_task(stack: List[str]) -> str:
    """
    Pop a task from a task stack.

    Args:
        stack (List[str]): The task stack.

    Returns:
        str
    """
    try:
        if stack:
            return stack.pop()
        else:
            return "Stack is empty."
    except Exception as e:
        return e
