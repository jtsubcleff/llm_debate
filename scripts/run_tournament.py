"""Run the centrally configured model roster with task-independent prompts."""
from pathlib import Path

import fire
import yaml

from core.llm_api.registry import catalog, model_id
from core.tournament import main as tournament_main
from core.utils import load_yaml


def build_roster(config_dir):
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    config = catalog()
    template = load_yaml("core/config/experiment/debaters/generic.yaml")
    debaters = []
    for selection in config["groups"]["debaters"]:
        player = {
            **template,
            "language_model": {**template["language_model"], "model": selection},
        }
        path = config_dir / f"{selection}.yaml"
        path.write_text(yaml.safe_dump(player, sort_keys=False))
        debaters.append((model_id(selection), path, {}))
    judges = [
        (
            model_id(selection),
            model_id(selection),
            Path("core/config/experiment/judge/debate/generic.yaml"),
        )
        for selection in config["groups"]["judges"]
    ]
    return debaters, judges


def main(tournament_dir: Path, method: str, anthropic_num_threads: int = 1):
    if method not in ("sp", "st"):
        raise ValueError(f"method {method} not supported")
    debaters, judges = build_roster(Path(tournament_dir) / "player_configs")
    tournament_main(
        tournament_dir=tournament_dir,
        debaters=debaters,
        judges=judges,
        match_up_type="self-play" if method == "sp" else "swiss-tournament",
        org="DEFAULT_ORG",
        limit=400 if method == "sp" else 291,
        num_threads=anthropic_num_threads,
        max_num_from_same_story=5,
        dataset_split="train" if method == "sp" else "dev",
    )


if __name__ == "__main__":
    fire.Fire(main)
