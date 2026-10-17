"""Model metadata and role resolution. No API IDs belong in runtime code.

Set LLM_MODELS_CONFIG to select an alternate catalog for an experiment.
"""

import argparse
import copy
import os
from functools import lru_cache
from pathlib import Path

import tiktoken
import yaml
from omegaconf import OmegaConf

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


@lru_cache(maxsize=8)
def _load(path: str):
    with open(path) as stream:
        config = yaml.safe_load(stream)
    for section in ("roles", "profiles", "models"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Model config requires a {section} mapping: {path}")
    return config


def catalog():
    return copy.deepcopy(
        _load(str(Path(os.environ.get("LLM_MODELS_CONFIG", DEFAULT_CONFIG)).resolve()))
    )


def resolve_model(name: str) -> dict:
    """Accept a role, catalog key, API ID, or explicit snapshot alias."""
    config = catalog()
    seen = set()
    while name in config["roles"]:
        if name in seen:
            raise ValueError(f"Cyclic model role: {name}")
        seen.add(name)
        name = config["roles"][name]
    entry = config["models"].get(name)
    if entry is None:
        entry = next(
            (
                value
                for value in config["models"].values()
                if name == value["id"] or name in value.get("aliases", [])
            ),
            None,
        )
    if entry is None:
        raise ValueError(
            f"Unknown model {name!r}. Add it to {DEFAULT_CONFIG} or LLM_MODELS_CONFIG."
        )
    spec = {**config["profiles"][entry["profile"]], **entry}
    # A requested snapshot must stay pinned rather than silently use the alias.
    if name in entry.get("aliases", []):
        spec["id"] = name
    if spec["provider"] not in ("openai", "anthropic"):
        raise ValueError(f"Unsupported provider: {spec['provider']}")
    endpoints = {
        "openai": ("chat", "responses", "completions"),
        "anthropic": ("messages", "completions"),
    }
    if spec["endpoint"] not in endpoints[spec["provider"]]:
        raise ValueError(
            f"Unsupported endpoint for {spec['provider']}: {spec['endpoint']}"
        )
    return spec


def model_id(name: str) -> str:
    return resolve_model(name)["id"]


def tokenizer(name: str):
    # SPECIAL TREATMENT: provider tokenizers differ. This local encoding is an
    # estimate for prompt budgeting; API usage is authoritative for billing.
    return tiktoken.get_encoding(resolve_model(name).get("tokenizer", "cl100k_base"))


def price_per_token(name: str):
    prices = resolve_model(name).get("price_per_million")
    if prices is None:
        return None
    return tuple(value / 1_000_000 for value in prices)


def estimate_cost(name: str, input_tokens: int, output_tokens: int):
    prices = price_per_token(name)
    return (
        None if prices is None else input_tokens * prices[0] + output_tokens * prices[1]
    )


OmegaConf.register_new_resolver("model", model_id, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role")
    parser.add_argument("--group")
    args = parser.parse_args()
    if args.role:
        print(model_id(args.role))
    elif args.group:
        print(" ".join(model_id(name) for name in catalog()["groups"][args.group]))
    else:
        parser.error("Choose --role or --group")


if __name__ == "__main__":
    main()
