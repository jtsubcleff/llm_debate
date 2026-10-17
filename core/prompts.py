"""Provider-independent templates; task adapters supply the context variables."""

import re
from pathlib import Path

import yaml

TEMPLATE_DIR = Path(__file__).resolve().parent / "config" / "prompts"


def load_template(name):
    path = TEMPLATE_DIR / f"{name}.yaml"
    if not path.exists():
        path = Path(name)
    with path.open() as stream:
        return yaml.safe_load(stream)


def render(content, variables):
    def substitute(match):
        key = match.group(1)
        if key not in variables:
            raise ValueError(f"Missing prompt variable: {key}")
        return str(variables[key])

    # Only uppercase placeholders are variables; XML output tags stay literal.
    return re.sub(r"<([A-Z][A-Z_0-9]*)>", substitute, content)
