import ast
from typing import Dict, List
import boto3
from tenacity import retry
from tools.utils import retry_settings

AWS_CLOUDFORMATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "aws_cloudformation_create_stack",
        "description": "Create an aws cloudformation stack given the name, template, and an optional list of stack's parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "StackName": {
                    "type": "string",
                    "description": "The name that's associated with the stack.",
                },
                "TemplateBody": {
                    "type": "string",
                    "description": "The valid template body of the stack.",
                },
                "Parameters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ParameterKey": {"type": "string"},
                            "ParameterValue": {"type": "string"},
                        },
                    },
                    "description": "A list of input parameters for the stack.",
                    "uniqueItems": True,
                },
            },
            "required": ["StackName", "TemplateBody"],
        },
    },
}


@retry(**retry_settings)
def aws_cloudformation_create_stack(
    StackName: str,
    TemplateBody: str,
    Parameters: str = [],
):
    with open("template.yaml", "w") as f:
        f.write(TemplateBody)

    # Initialize boto3 client
    client = boto3.client("cloudformation")

    # Create stack
    response = client.create_stack(
        StackName=StackName,
        TemplateBody=TemplateBody,
        Capabilities=[
            "CAPABILITY_IAM",
            "CAPABILITY_NAMED_IAM",
            "CAPABILITY_AUTO_EXPAND",
        ],
        Parameters=Parameters,
    )

    # Wait for stack creation to complete
    waiter = client.get_waiter("stack_create_complete")
    waiter.wait(StackName=StackName)

    return response
