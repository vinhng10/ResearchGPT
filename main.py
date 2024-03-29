from agent import Agent
from tools import (
    READ_FILE_SCHEMA,
    WRITE_TO_FILE_SCHEMA,
    GET_DIRECTORY_TREE_SCHEMA,
    read_file,
    write_to_file,
    get_directory_tree,
)


if __name__ == "__main__":
    agent = Agent(
        system="""You are an experienced software engineer. Given a description of an application, you will analyze its functional and technical requirements, from there break down the application into smaller components and tasks You will gradually write code for each task and save the implementation into files. The project start at the directory called 'root'. You have a task stack to store what tasks to be done. Please push tasks to the stack while planning, and pop tasks from there for execution until stack is empty.""",
        tool_schemas=[
            READ_FILE_SCHEMA,
            WRITE_TO_FILE_SCHEMA,
            GET_DIRECTORY_TREE_SCHEMA,
        ],
        tools={
            "read_file": read_file,
            "write_to_file": write_to_file,
            "get_directory_tree": get_directory_tree,
        },
    )

    agent.chat(
        "Write an e-commerce platform that allows users to sell their products online."
    )
