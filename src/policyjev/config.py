"""Project-wide configuration constants and loader."""

from pathlib import Path

import yaml

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    """Load configs/project.yaml as a dict."""
    config_path = REPO_ROOT / "configs" / "project.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


LABELS: list[str] = load_config()["labels"]
