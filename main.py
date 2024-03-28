from agent import Agent
from tools import (
    WRITE_TO_FILE_SCHEMA,
    GET_DIRECTORY_TREE_SCHEMA,
    write_to_file,
    get_directory_tree,
)


if __name__ == "__main__":
    stack = []
    agent = Agent(
        system="""You are an experienced software engineer. Given a description of an application, you will analyze its functional and technical requirements, from there break down the application into smaller components and tasks. You will gradually write code for each task and save the implementation into files.""",
        tool_schemas=[
            WRITE_TO_FILE_SCHEMA,
            GET_DIRECTORY_TREE_SCHEMA,
        ],
        tools={
            "write_to_file": write_to_file,
            "get_directory_tree": get_directory_tree,
        },
    )

    agent.chat(
        "Write an e-commerce platform that allows users to sell their products online."
    )

