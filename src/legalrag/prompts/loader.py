"""Prompt template loader and renderer."""

from pathlib import Path


def get_prompts_dir() -> Path:
    """Return the absolute path to prompts root directory."""
    return Path(__file__).parent


def load_prompt(name: str, version: str = "v1") -> str:
    """Load raw markdown prompt template by name and version."""
    prompt_path = get_prompts_dir() / version / f"{name}.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found at {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def render_prompt(template: str, **kwargs: str) -> str:
    """Format prompt template with provided keyword arguments safely."""
    # Replace kwargs safely without crashing on unescaped brackets
    text = template
    for k, v in kwargs.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text
