from io import StringIO
import sys
from directory_tree import display_tree
from pathlib import Path
from tenacity import retry
from .utils import pretty_print, retry_settings


READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read content of a file specified by the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file.",
                },
            },
            "required": ["path"],
        },
    },
}


WRITE_TO_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_to_file",
        "description": "Write content to a file specified by the given path. If the file does not exist, the function creates it along with any necessary parent directories. Otherwise, the file's content is overwritten.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file.",
                },
                "content": {
                    "type": "string",
                    "description": "The content of the file.",
                },
            },
            "required": ["path", "content"],
        },
    },
}


GET_DIRECTORY_TREE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_directory_tree",
        "description": "Generate a directory tree structure for the specified directory path.",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The path to the directory.",
                },
            },
            "required": ["directory"],
        },
    },
}


@retry(**retry_settings)
@pretty_print(color="green")
def read_file(path: str) -> str:
    """
    Read content of a file specified by the given path.

    Args:
        path (str): The path to the file to be read.

    Returns:
        str
    """
    try:
        # Open the file in read mode:
        with open(path, "r") as file:
            content = file.read()
        return content

    except IOError as e:
        return e


@retry(**retry_settings)
@pretty_print(color="green")
def write_to_file(path: str, content: str) -> str:
    """
    Write content to a file specified by the given path. If the file does not exist,
    the function creates it along with any necessary parent directories. If the file
    already exists, its content is overwritten.

    Args:
        path (str): The path to the file to be written.
        content (str): The content to be written to the file.

    Returns:
        str
    """
    try:
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Open the file in write mode
        with open(path, "w") as file:
            # Write the content to the file
            file.write(content)

        return f"Content written to {path}"

    except Exception as e:
        return e


@retry(**retry_settings)
@pretty_print(color="cyan")
def get_directory_tree(directory: str) -> str:
    """
    Generate a directory tree structure for the specified directory path.

    Args:
        directory (str): The path to the directory.

    Returns:
        str
    """
    try:
        sys.stdout = StringIO()
        tree = display_tree(dir_path=directory, string_rep=True)
        exception = sys.stdout.getvalue()
        return tree if tree else exception
    except Exception as e:
        return e
