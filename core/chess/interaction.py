"""Checkpointed chess interaction experiments; final judging runs separately."""

import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

from core.chess.protocols.direct_qa import prompt_template_name
from core.chess.protocols.interaction import run_interaction, turn_schedule
from core.chess.results import aggregate_results, write_csv, write_json
from core.chess.tools import CHESS_TOOLS, TOOL_COUNT_FIELDS
from core.llm_api.registry import resolve_model
from core.load.chess import load_puzzles, sample_puzzles
from core.prompts import load_template

LOGGER = logging.getLogger(__name__)


def interaction_models(cfg):
    names = [cfg.models] if isinstance(cfg.models, str) else list(cfg.models)
    if not names:
        raise ValueError("Specify at least one model or consultancy pair")
    jobs = {}
    for entry in names:
        if cfg.protocol == "consultancy":
            if isinstance(entry, str) or len(entry) != 2:
                raise ValueError(
                    "consultancy models must be [proposer, questioner] pairs"
                )
            proposer, questioner = [resolve_model(name) for name in entry]
        else:
            if not isinstance(entry, str):
                raise ValueError("debate models must be a list of model names")
            proposer = questioner = resolve_model(entry)
        if any(s["endpoint"] == "completions" for s in (proposer, questioner)):
            raise ValueError("Chess requires Chat, Responses, or Messages models")
        jobs[(proposer["id"], questioner["id"])] = (proposer, questioner)
    return jobs


async def run_interactions(cfg, api):
    from core.chess.run import model_directory

    turn_schedule(cfg.protocol, cfg.num_rounds)
    if cfg.max_tokens <= 0 or cfg.max_tool_rounds < 0 or cfg.num_threads <= 0:
        raise ValueError(
            "Require max_tokens > 0, max_tool_rounds >= 0, num_threads > 0"
        )
    jobs = interaction_models(cfg)
    puzzles = sample_puzzles(load_puzzles(Path(cfg.dataset_path)), cfg.limit, cfg.seed)
    exp_dir = Path(cfg.exp_dir)
    settings = dict(
        protocol=cfg.protocol,
        transcript_content="public_text_only_v1",
        num_rounds=cfg.num_rounds,
        use_tools=cfg.use_tools,
        seed=cfg.seed,
        limit=cfg.limit,
        max_tokens=cfg.max_tokens,
        max_tool_rounds=cfg.max_tool_rounds,
        temperature=cfg.temperature,
        prompt=load_template(prompt_template_name(cfg.use_tools)),
        interaction_prompt=load_template("chess_interaction"),
        tool=CHESS_TOOLS if cfg.use_tools else None,
        puzzles=[p.to_dict() for p in puzzles],
    )
    manifest = exp_dir / "chess_run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != settings:
        raise ValueError("Experiment settings or sample changed; choose a new exp_dir")
    frames, files, pending, profiles = {}, {}, [], []
    for key, specs in jobs.items():
        model, questioner = key
        name = model_directory(model)
        if cfg.protocol == "consultancy":
            name += "__" + model_directory(questioner)
        folder = exp_dir / cfg.protocol / name
        profile_path = folder / "model.json"
        profile = dict(
            proposer=specs[0],
            **{"questioner" if cfg.protocol == "consultancy" else "critic": specs[1]},
        )
        if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
            raise ValueError("Model profile changed; choose a new exp_dir")
        profiles.append((profile_path, profile))
        path = folder / "puzzles.csv"
        if path.exists():
            frame = pd.read_csv(
                path, keep_default_na=False, dtype={"puzzleId": str}
            ).astype(object)
            if frame.puzzleId.tolist() != [p.puzzleId for p in puzzles]:
                raise ValueError("Saved sample differs; choose a new exp_dir")
        else:
            frame = pd.DataFrame(
                [
                    dict(
                        **p.to_dict(),
                        **dict.fromkeys(TOOL_COUNT_FIELDS, 0),
                        model=model,
                        questioner=questioner if cfg.protocol == "consultancy" else "",
                        use_tools=cfg.use_tools,
                        num_rounds=cfg.num_rounds,
                        complete=False,
                        correct=False,
                        final_move="",
                        model_response="",
                        transcript="[]",
                        tool_trace="[]",
                        error="",
                        request_id="",
                    )
                    for p in puzzles
                ]
            ).astype(object)
        frames[key], files[key] = frame, path
        pending.extend(
            (key, index, puzzle, folder / "cache")
            for index, puzzle in enumerate(puzzles)
            if str(frame.at[index, "complete"]).lower() != "true"
        )
    write_json(manifest, settings)
    for path, profile in profiles:
        write_json(path, profile)
    for key, path in files.items():
        write_csv(frames[key], path)
    aggregate_results(exp_dir, cfg.protocol)
    semaphore = asyncio.Semaphore(cfg.num_threads)

    async def evaluate(key, index, puzzle, cache):
        async with semaphore:
            result = await run_interaction(
                api,
                key[0],
                puzzle,
                cache,
                cfg.protocol,
                cfg.num_rounds,
                questioner_model=key[1],
                max_tokens=cfg.max_tokens,
                max_tool_rounds=cfg.max_tool_rounds,
                temperature=cfg.temperature,
                use_tools=cfg.use_tools,
            )
            frame = frames[key]
            for field, value in result.items():
                if field not in frame:
                    frame[field] = pd.Series([None] * len(frame), dtype=object)
                frame.at[index, field] = value
            write_csv(frame, files[key])
            aggregate_results(exp_dir, cfg.protocol)
            LOGGER.info(
                "%s %s puzzle %s: complete=%s correct=%s",
                cfg.protocol,
                key,
                puzzle.puzzleId,
                result["complete"],
                result["correct"],
            )
            return result

    results = await asyncio.gather(*(evaluate(*item) for item in pending))
    if any(not result["complete"] for result in results):
        raise RuntimeError(
            "Chess interactions failed; details saved in puzzles.csv. Rerun the same command to resume."
        )
    return frames
