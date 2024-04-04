from agent import Agent
from tools import (
    READ_FILE_SCHEMA,
    RUN_COMMAND_SCHEMA,
    read_file,
    run_command,
)


if __name__ == "__main__":
    agent = Agent(
        system="""You are an experienced software engineer. \n\nGiven an application description, you will either plan what tasks to do or write code to fulfill a task.\n\nPick only one of the following formats for your response.\n\nIf you want to plan your work by tasks, use only the following format:\nTYPE: PLAN\n**********\nTASK:\n{task detailed description}\n===\nTASK:\n{task detailed description}\n\nIf you want to write code to fulfill the task, use only the following format:\nTYPE: CODE\n**********\nFILE:\n{file path}\n===\nCODE:\n{code}\n\nAlso utilize the given tools if necessary to help you finish the project.\n""",
        tool_schemas=[READ_FILE_SCHEMA, RUN_COMMAND_SCHEMA],
        tools={"read_file": read_file, "run_command": run_command},
    )

    agent.conversation(
        "Create a platform that let users trade stocks in real-time. Use React.js and AWS for implementation. Start the project at directory called 'root'."
    )
