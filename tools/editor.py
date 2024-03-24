import logging
from typing import Any
from directory_tree import display_tree
from pathlib import Path
from tenacity import retry
from utils import retry_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def write_to_file(path: str, content: str) -> str:
    """
    Write content to a file specified by the given path. If the file does not exist,
    the function creates it along with any necessary parent directories. If the file
    already exists, its content is overwritten.

    Args:
        path (str): The path to the file to be written.
        content (str): The content to be written to the file.

    Returns:
        None

    Raises:
        IOError: If there is an error writing to the file.
    """
    try:
        # Create parent directories if they don't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Open the file in write mode
        with open(path, "w") as file:
            # Write the content to the file
            file.write(content)

        logger.info(f"Content written to {path}")
        return

    except IOError as e:
        logger.error(f"Error writing to file: {e}")


@retry(**retry_settings)
def get_directory_tree(directory: str) -> str | Any | None:
    """
    Generate a directory tree structure for the specified directory path.

    Args:
        directory (str): The path to the directory.

    Returns:
        str | Any | None
    """
    import sys
    from io import StringIO

    sys.stdout = StringIO()
    tree = display_tree(dir_path=directory, string_rep=True)
    exception = sys.stdout.getvalue()
    return tree if tree else exception
