from typing import List
from tenacity import retry
from utils import retry_settings


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
def push_task(task: str, stack: List[str]) -> None:
    """
    Push a task to a task stack.

    Args:
        task (str): The task.
        stack (List[str]): The task stack.

    Returns:
        None
    """
    stack.append(task)


@retry(**retry_settings)
def pop_task(stack: List[str]) -> str:
    """
    Pop a task from a task stack.

    Args:
        stack (List[str]): The task stack.

    Returns:
        str
    """
    return stack.pop()
