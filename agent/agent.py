import json
from pathlib import Path
from typing import Any, Callable, Dict, List
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from tenacity import retry
from termcolor import colored
from tools import (
    retry_settings,
    write_to_file,
)


class Agent:
    def __init__(
        self,
        system: str,
        tool_schemas: List[Dict],
        tools: Dict[str, Callable[..., Any]],
        model: str = "gpt-3.5-turbo-0125",
    ) -> None:
        super().__init__()
        Path("root").mkdir(exist_ok=True, parents=True)
        self.model = model
        self.tool_schemas = tool_schemas
        self.tools = tools
        self.llm = OpenAI()
        self.system = [
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": "APP:\nCreate a basic counter app featuring an increment and a decrement button. Clicking the increment button increases the displayed number by 1, while clicking the decrement button decreases it by 1. Utilize HTML, CSS, and JavaScript for implementation.",
            },
            {
                "role": "assistant",
                "content": "I received your request. I will start working on it.",
            },
            {
                "role": "user",
                "content": "TASK:\nDraft initial plan to build the application.",
            },
            {
                "role": "assistant",
                "content": "TYPE: PLAN\n**********\nTASK:\nWrite an HTML file that contains the structure of the web page.\n===\nTASK:\nWrite a CSS file that contains the styles to make the app look better.\n===\nTASK:\nDevelop JavaScript code that increments the counter upon clicking the increment button.\n===\nTASK:\nDevelop JavaScript code that decrements the counter upon clicking the decrement button.\n===\nTASK:\nRun the app with Python webserver.",
            },
            {
                "role": "user",
                "content": "TASK:\nWrite an HTML file that contains the structure of our web page.",
            },
            {
                "role": "assistant",
                "content": 'TYPE: PLAN\n**********\nFILE:\nroot/index.html\n===\nCODE:\n<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <meta charset="UTF-8" />\n    <meta name="viewport" content="width=device-width, initial-scale=1.0" />\n    <title>Counter App</title>\n    <link rel="stylesheet" href="styles.css" />\n  </head>\n  <body>\n    <div class="counter">\n      <h1 id="count">0</h1>\n      <button id="incrementBtn">Increment</button>\n      <button id="decrementBtn">Decrement</button>\n    </div>\n    <script src="script.js"></script>\n  </body>\n</html>\n',
            },
            {
                "role": "user",
                "content": "TASK:\nWrite a CSS file that contains the styles to make the app look better.",
            },
            {
                "role": "assistant",
                "content": "TYPE: PLAN\n**********\nFILE:\nroot/styles.css\n===\nCODE:\nbody {\n  font-family: Arial, sans-serif;\n  display: flex;\n  justify-content: center;\n  align-items: center;\n  height: 100vh;\n  margin: 0;\n}\n\n.counter {\n  text-align: center;\n}\n\nh1 {\n  font-size: 3rem;\n}\n\nbutton {\n  padding: 10px 20px;\n  font-size: 1.2rem;\n  cursor: pointer;\n}",
            },
            {
                "role": "user",
                "content": "TASK:\nDevelop JavaScript code that increments the counter upon clicking the increment button.",
            },
            {
                "role": "assistant",
                "content": 'TYPE: PLAN\n**********\nFILE:\nroot/script.js\n===\nCODE:\ndocument.addEventListener("DOMContentLoaded", function () {\n  const countElement = document.getElementById("count");\n  const incrementButton = document.getElementById("incrementBtn");\n  let count = 0;\n\n  incrementButton.addEventListener("click", function () {\n    count++;\n    countElement.textContent = count;\n  });\n});\n',
            },
            {
                "role": "user",
                "content": "TASK:\nDevelop JavaScript code that decrements the counter upon clicking the decrement button.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        type="function",
                        id="call_000000000000000000000000",
                        function=Function(
                            name="get_directory_tree",
                            arguments='{"directory": "root"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_000000000000000000000000",
                "name": "get_directory_tree",
                "content": "root/\n├── index.html\n└── styles.css\n└── script.js",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        type="function",
                        id="call_000000000000000000000001",
                        function=Function(
                            name="read_file",
                            arguments='{"path": "root/script.js"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_000000000000000000000001",
                "name": "read_file",
                "content": 'document.addEventListener("DOMContentLoaded", function () {\n  const countElement = document.getElementById("count");\n  const incrementButton = document.getElementById("incrementBtn");\n  let count = 0;\n\n  incrementButton.addEventListener("click", function () {\n    count++;\n    countElement.textContent = count;\n  });\n});\n',
            },
            {
                "role": "assistant",
                "content": 'TYPE: PLAN\n**********\nFILE:\nroot/script.js\n===\nCODE:\ndocument.addEventListener("DOMContentLoaded", function () {\n  const countElement = document.getElementById("count");\n  const incrementButton = document.getElementById("incrementBtn");\n  const decrementButton = document.getElementById("decrementBtn");\n  let count = 0;\n\n  incrementButton.addEventListener("click", function () {\n    count++;\n    countElement.textContent = count;\n  });\n\n  decrementButton.addEventListener("click", function () {\n    count--;\n    countElement.textContent = count;\n  });\n});\n',
            },
            {
                "role": "user",
                "content": "TASK:\nRun the app with Python webserver.",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    ChatCompletionMessageToolCall(
                        type="function",
                        id="call_000000000000000000000002",
                        function=Function(
                            name="run_command",
                            arguments='{"command": "cd root && python3 -m http.server"}',
                        ),
                    )
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_000000000000000000000002",
                "name": "run_command",
                "content": "",
            },
            {
                "role": "assistant",
                "content": "DONE",
            },
        ]
        self.messages = []
        self.stack = []

    @retry(**retry_settings)
    def push(self, task: str) -> str:
        try:
            self.stack.append(task)
            return task
        except Exception as e:
            return e

    @retry(**retry_settings)
    def pop(self) -> str:
        try:
            if self.stack:
                return self.stack.pop()
            else:
                return "Stack is empty."
        except Exception as e:
            return e

    def chat(self, message: str) -> None:
        self.messages.append({"role": "user", "content": message})
        while True:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=self.system + self.messages,
                tools=self.tool_schemas,
                tool_choice="auto",
            )
            choice = response.choices[0]
            message = choice.message

            # Finish reason:
            if choice.finish_reason in ["stop", "length", "content_filter"]:
                self.messages.append({"role": message.role, "content": message.content})
                break

            self.messages.append(message)

            tool_calls = message.tool_calls
            if tool_calls:
                for tool_call in tool_calls:
                    self.use_tool(tool_call)

        self.messages = []
        return message.content

    def conversation(self, initial_message: str) -> None:
        self.system = self.system + [
            {
                "role": "user",
                "content": f"APP:\n{initial_message}",
            },
            {
                "role": "assistant",
                "content": "I received your request. I will start working on it.",
            },
        ]
        self.push("TASK:\nDraft initial plan to build the application.")
        while self.stack:
            task = self.pop()
            print(colored(f"\n{task}\n", "red"))

            message = self.chat(task)
            parts = message.split("**********")

            type = parts[0].strip()
            if type == "TYPE: PLAN":
                print(colored(message, "cyan"))
                for task in parts[1].split("===")[::-1]:
                    self.push(task.strip())
            elif type == "TYPE: CODE":
                path, code = parts[1].split("===")
                path = path[7:].strip()
                code = code[7:]
                write_to_file(path, code)
            else:
                print(colored(message, "yellow"))

    def use_tool(self, tool_call: ChatCompletionMessageToolCall) -> None:
        try:
            tool_name = tool_call.function.name
            tool = self.tools[tool_name]
            arguments = json.loads(tool_call.function.arguments)
            response = tool(**arguments)
        except KeyError as e:
            response = f'Tool "{e}" is not available.'
        except Exception as e:
            response = str(e)
        finally:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": json.dumps(response),
                }
            )
