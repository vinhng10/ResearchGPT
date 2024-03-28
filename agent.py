import json
from functools import partial
from typing import Any, Callable, Dict, List
from openai import OpenAI
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from termcolor import colored
from tools import PUSH_TASK_SCHEMA, POP_TASK_SCHEMA, push_task, pop_task


class Agent:
    def __init__(
        self,
        system: str,
        tool_schemas: List[Dict],
        tools: Dict[str, Callable[..., Any]],
        model: str = "gpt-3.5-turbo-0125",
    ) -> None:
        super().__init__()
        self.stack = []
        self.model = model
        self.tool_schemas = tool_schemas
        self.tool_schemas.extend([PUSH_TASK_SCHEMA, POP_TASK_SCHEMA])
        self.tools = tools
        self.tools.update(
            {
                "push_task": partial(push_task, stack=self.stack),
                "pop_task": partial(pop_task, stack=self.stack),
            }
        )
        self.llm = OpenAI()
        self.messages = [
            {
                "role": "system",
                "content": system,
            }
        ]

    def pretty_print(func, color: str) -> Callable[..., Any]:
        def wrapper(self, *args, **kwargs) -> Any:
            output = func(self, *args, **kwargs)
            print(colored(output, color))
            return output

        return wrapper

    # @partial(pretty_print, color="cyan")
    def chat(self, message: str) -> ChatCompletionMessage:
        self.messages.append({"role": "user", "content": message})
        while True:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tool_schemas,
                tool_choice="auto",
            )
            choice = response.choices[0]

            # Finish reason:
            if choice.finish_reason in ["stop", "length", "content_filter"]:
                break

            message = choice.message
            self.messages.append(message)

            tool_calls = message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    self.use_tool(tool_call)

    @partial(pretty_print, color="green")
    def use_tool(self, tool_call) -> str:
        tool_name = tool_call.function.name
        tool = self.tools[tool_name]
        arguments = json.loads(tool_call.function.arguments)
        response = tool(**arguments)
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(response),
            }
        )
        return f"TOOL: {tool_name}\nRESULT: {json.dumps(response)}\n"
