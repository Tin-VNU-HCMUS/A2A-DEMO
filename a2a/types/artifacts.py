# a2a/types/artifacts.py
from pydantic import BaseModel, validator
from typing import Optional

class TextArtifact(BaseModel):
    """
    A Pydantic model representing a text artifact with type and content.
    
    Attributes:
        type (str): The type of the artifact, defaults to "text".
        text (str): The content of the text artifact.
    """
    type: str = "text"
    text: str

    @validator('text')
    def text_must_not_be_empty(cls, v):
        """Validate that text is not empty or whitespace only."""
        if not v or v.isspace():
            raise ValueError("Text cannot be empty or consist only of whitespace.")
        return v

    def to_dict(self) -> dict:
        """Convert the artifact to a dictionary."""
        return {"type": self.type, "text": self.text}

    def __str__(self) -> str:
        """Return a string representation of the artifact."""
        return f"TextArtifact(type={self.type}, text={self.text})"

if __name__ == "__main__":
    # Example usage
    try:
        artifact = TextArtifact(text="Hello, world!")
        print(artifact.to_dict())
    except ValueError as e:
        print(f"Error: {e}")



