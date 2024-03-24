from typing import Dict

from tenacity import stop_after_attempt, wait_exponential
from termcolor import colored


# Retry decorator settings
retry_settings = {
    "wait": wait_exponential(multiplier=1, min=4, max=10),
    "stop": stop_after_attempt(3),
}


def pretty_print(message: Dict[str, str]) -> None:
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }

    if message["role"] == "system":
        print(
            colored(f"SYSTEM: {message['content']}\n", role_to_color[message["role"]])
        )
    elif message["role"] == "user":
        print(colored(f"USER: {message['content']}\n", role_to_color[message["role"]]))
    elif message["role"] == "assistant" and message.get("tool_calls"):
        print(
            colored(
                f"ASSISTANT: {message['tool_calls']}\n",
                role_to_color[message["role"]],
            )
        )
    elif message["role"] == "assistant" and not message.get("tool_calls"):
        print(
            colored(
                f"ASSISTANT: {message['content']}\n", role_to_color[message["role"]]
            )
        )
    elif message["role"] == "tool":
        print(
            colored(
                f"TOOL: ({message['name']}): {message['content']}\n",
                role_to_color[message["role"]],
            )
        )
