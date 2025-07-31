from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .prompt import Prompt


class PromptLoader:
    """
    A class to load prompts from YAML files.

    Attributes:
        prompts_dir (Path): Directory containing prompt YAML files
        loaded_prompts (Dict[Tuple[str, Optional[str]], Prompt]): Cache of loaded prompts
    """

    def __init__(self, prompts_dir: Optional[str] = None) -> None:
        """
        Initialize the PromptLoader.

        Args:
            prompts_dir (str): Path to the prompts directory.
        """
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.loaded_prompts: Dict[str, Prompt] = {}

    def load_prompt(self, name: str) -> Prompt:
        """
        Load a prompt by name.

        Args:
            name (str): Name of the prompt file (without extension)

        Returns:
            Prompt: The loaded prompt configuration

        Raises:
            FileNotFoundError: If prompt file not found
        """
        cache_key = name
        if cache_key in self.loaded_prompts:
            return self.loaded_prompts[cache_key]

        if self.prompts_dir is None:
            raise ValueError("No prompts directory specified")

        prompt_file = self.prompts_dir / f"{name}.yaml"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file {prompt_file} not found")

        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_data = yaml.safe_load(f)

        prompt = Prompt.from_config(prompt_data)
        self.loaded_prompts[cache_key] = prompt
        return prompt

    def list_available_prompts(self) -> List[str]:
        """
        List all available prompts in the prompts directory.

        Returns:
            List[str]: List of prompt names (without extension)
        """
        if self.prompts_dir is None:
            raise ValueError("No prompts directory specified")

        return [f.stem for f in self.prompts_dir.glob("*.yaml")]
