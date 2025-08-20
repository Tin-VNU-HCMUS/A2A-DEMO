# File: a2a/utils/message_factory.py
from a2a.types.artifacts import TextArtifact
from typing import Optional

def new_agent_text_message(text: str, parent_id: Optional[str] = None) -> dict:
    """
    Create a new text message for an agent.
    
    Args:
        text (str): The content of the message.
        parent_id (Optional[str]): The ID of the parent message, if any.
    
    Returns:
        dict: A dictionary containing message details.
    
    Raises:
        ValueError: If text is empty or not a string.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text must be a non-empty string.")
    
    return {
        "type": "text",
        "text": text,
        "parent_id": parent_id
    }

def create_text_artifact(message: dict) -> TextArtifact:
    """
    Convert a message dictionary to a TextArtifact instance.
    
    Args:
        message (dict): The message dictionary from new_agent_text_message.
    
    Returns:
        TextArtifact: A Pydantic model instance.
    """
    return TextArtifact(type=message.get("type", "text"), text=message["text"])

if __name__ == "__main__":
    # Example usage
    try:
        msg = new_agent_text_message("Hello, this is a test message!", "parent123")
        artifact = create_text_artifact(msg)
        print(artifact.json())
    except ValueError as e:
        print(f"Error: {e}")