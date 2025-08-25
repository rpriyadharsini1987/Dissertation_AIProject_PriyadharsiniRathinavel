import json
import os

PROMPT_FILE_PATH = os.path.join("prompts", "2_prompt.json")

class PromptManager:
    def __init__(self, prompt_file_path: str = PROMPT_FILE_PATH):
        if not os.path.exists(prompt_file_path):
            raise FileNotFoundError(f"Prompt file '{prompt_file_path}' not found.")

        with open(prompt_file_path, 'r', encoding='utf-8') as f:
            try:
                self.prompts = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in prompt file: {e}")

    def get(self, key: str, **kwargs) -> str:
        if key not in self.prompts:
            raise KeyError(f"Prompt key '{key}' not found in prompt file.")

        prompt_template = self.prompts[key]
        try:
            return prompt_template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Missing placeholder for prompt '{key}': {e}")

# Optional: Quick test
if __name__ == "__main__":
    pm = PromptManager()
    sample = pm.get("generate_keywords", topic="Encapsulation")
    print(sample)
