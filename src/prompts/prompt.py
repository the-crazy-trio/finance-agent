from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class Prompt(BaseModel):
    """
    Represents a loaded prompt with its configuration.

    Attributes:
        system (str): System message for the LLM
        user (str): User message template
        text (str): Alias for 'user' or prompt text (backwards compatibility)
        name (str): Name of the prompt
        description (str): Description of what the prompt does
        response_format (Optional[Dict]): Response format configuration for the API
    """

    system: Optional[str] = Field(None, description="System message for the LLM")
    user: Optional[str] = Field(None, description="User message template")
    text: Optional[str] = Field(
        None, description="Prompt text (backwards compatibility)"
    )
    name: str = Field(..., description="Name of the prompt")
    description: str = Field(..., description="Description of what the prompt does")
    response_format: Optional[Dict] = Field(
        None, description="Response format configuration for the API"
    )

    @model_validator(mode="before")
    def handle_text_user_compatibility(cls, values):
        """Support both 'text' and 'user' fields for compatibility."""
        # If prompt is in the values, it's likely from older test fixture
        if "prompt" in values and not values.get("user") and not values.get("text"):
            values["text"] = values.pop("prompt")

        # Handle the case where only one of text/user is provided
        if values.get("text") is not None and values.get("user") is None:
            values["user"] = values["text"]
        elif values.get("user") is not None and values.get("text") is None:
            values["text"] = values["user"]

        return values

    @classmethod
    def from_config(cls, config: Dict) -> "Prompt":
        """
        Create a Prompt instance from a configuration dictionary.

        Args:
            config (Dict): Configuration dictionary loaded from YAML
                Must contain: name, description
                Optional: system, user, text, response_format

        Returns:
            Prompt: A new Prompt instance

        Raises:
            ValueError: If required fields are missing from config
        """
        try:
            # Handle backward compatibility with prompt vs user/text
            if "prompt" in config and "user" not in config and "text" not in config:
                config["text"] = config.pop("prompt")

            return cls(
                system=config.get("system", None),
                user=config.get("user", config.get("text", None)),
                text=config.get("text", config.get("user", None)),
                name=config["name"],
                description=config["description"],
                response_format=config.get("response_format"),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in config: {e.args[0]}")

    def format(self, **kwargs) -> "Prompt":
        """
        Format the prompt text with the given arguments.

        Args:
            **kwargs: Arguments to format the prompt text with

        Returns:
            Prompt: A new Prompt instance with formatted text
        """
        user_text = self.user or self.text
        if user_text is None:
            raise ValueError("No prompt text available to format")

        formatted_text = user_text.format(**kwargs)
        return Prompt(
            system=self.system,
            user=formatted_text,
            text=formatted_text,
            name=self.name,
            description=self.description,
            response_format=self.response_format,
        )

    def to_messages(self) -> List[Dict[str, str]]:
        """
        Convert the prompt to a list of messages for the API.

        Returns:
            List[Dict[str, str]]: List of messages in the format expected by the API
        """
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})

        user_content = self.user or self.text
        if user_content:
            messages.append({"role": "user", "content": user_content})
        return messages
