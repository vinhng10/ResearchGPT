from typing import Any, Callable, Dict, List
from openai import OpenAI
from termcolor import colored
import json

from tools import *

client = OpenAI()


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


def run_conversation(
    prompt: str, tool_shemas: List[Dict[str, Any]], tools: Dict[str, Callable]
) -> None:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that will answer any question. If you are unsure about the given topic, use available tools to gain more information and knowledge before answering.",
        },
        {"role": "user", "content": prompt},
    ]
    [pretty_print(message) for message in messages]
    stop = False

    while not stop:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            tools=tool_shemas,
            tool_choice="auto",
        )

        # Check the stopping reason
        stop = response.choices[0].finish_reason == "stop"

        message = response.choices[0].message
        messages.append(
            {
                "role": message.role,
                "content": message.content,
                "tool_calls": message.tool_calls,
            }
        )
        pretty_print(messages[-1])

        # Use tools:
        tool_calls = message.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": json.dumps(function_response),
                    }
                )
                pretty_print(messages[-1])


run_conversation(
    "Give me a report of timeline of events in Russia-Ukraine up to date. Don't just give me link. Research the topic, use the available tools whenever necessary, and make a report to me.",
    [GOOGLE_SEARCH_TOOL, PARSE_TEXTS_TOOL, PARSE_ANCHORS_TOOL],
    {
        "google_search": google_search,
        "parse_texts_from_webpage": parse_texts_from_webpage,
        "parse_anchors_from_webpage": parse_anchors_from_webpage,
    },
)
