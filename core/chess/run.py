"""Run one chess protocol across a shared balanced sample and multiple models."""

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from core.chess.protocols.direct_qa import prompt_template_name, run_direct_qa
from core.chess.results import aggregate_results, print_run_cost, write_csv, write_json
from core.chess.tools import CHESS_TOOLS, TOOL_COUNT_FIELDS
from core.llm_api.llm import ModelAPI
from core.llm_api.registry import resolve_model
from core.load.chess import load_puzzles, sample_puzzles
from core.prompts import load_template
from core.utils import setup_environment

LOGGER = logging.getLogger(__name__)
PROTOCOLS = {"direct_qa": run_direct_qa}


def model_directory(model):
    return (
        re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        + "_"
        + hashlib.sha256(model.encode()).hexdigest()[:8]
    )


async def run_experiment(cfg, api):
    try:
        return await _run_experiment(cfg, api)
    finally:
        print_run_cost(cfg.exp_dir, cfg.protocol)


async def _run_experiment(cfg, api):
    if cfg.protocol in ("debate", "consultancy"):
        from core.chess.interaction import run_interactions

        return await run_interactions(cfg, api)
    if cfg.protocol == "judge":
        from core.chess.judge import run_judging

        return await run_judging(cfg, api)
    if cfg.protocol == "proposer_judge":
        return await run_proposer_judge(cfg, api)
    if cfg.protocol not in PROTOCOLS:
        raise ValueError(f"Unknown chess protocol {cfg.protocol!r}")
    if cfg.max_tokens <= 0 or cfg.max_tool_rounds < 0 or cfg.num_threads <= 0:
        raise ValueError(
            "Require max_tokens > 0, max_tool_rounds >= 0, num_threads > 0"
        )
    names = [cfg.models] if isinstance(cfg.models, str) else list(cfg.models)
    if not names:
        raise ValueError("Specify at least one model")
    specs = [resolve_model(name) for name in names]
    if any(spec["endpoint"] == "completions" for spec in specs):
        raise ValueError(
            "Chess tool use requires Chat, Responses, or Messages models; legacy Completions is unsupported"
        )
    puzzles = sample_puzzles(load_puzzles(Path(cfg.dataset_path)), cfg.limit, cfg.seed)
    exp_dir = Path(cfg.exp_dir)
    # Pin the exact sample, model profiles, prompts, and generation settings.
    # Adding another model is supported; changing a run's semantics is rejected.
    settings = dict(
        protocol=cfg.protocol,
        use_tools=cfg.use_tools,
        seed=cfg.seed,
        limit=cfg.limit,
        max_tokens=cfg.max_tokens,
        max_tool_rounds=cfg.max_tool_rounds,
        temperature=cfg.temperature,
        prompt=load_template(prompt_template_name(cfg.use_tools)),
        tool=CHESS_TOOLS if cfg.use_tools else None,
        puzzles=[p.to_dict() for p in puzzles],
    )
    manifest = exp_dir / "chess_run.json"
    if manifest.exists():
        saved_settings = json.loads(manifest.read_text())
        # Runs created before this flag always enabled tools.
        saved_settings.setdefault("use_tools", True)
        if saved_settings != settings:
            raise ValueError(
                "Experiment settings or sample changed; choose a new exp_dir"
            )
    frames, files, pending = {}, {}, []
    profiles = []
    for model, spec in zip([s["id"] for s in specs], specs):
        if model in frames:
            continue
        folder = exp_dir / cfg.protocol / model_directory(model)
        profile_path = folder / "model.json"
        if profile_path.exists() and json.loads(profile_path.read_text()) != spec:
            raise ValueError(f"Model profile changed for {model}; choose a new exp_dir")
        profiles.append((profile_path, spec))
        path = folder / "puzzles.csv"
        if path.exists():
            frame = pd.read_csv(path, keep_default_na=False, dtype={"puzzleId": str})
            if frame.puzzleId.tolist() != [p.puzzleId for p in puzzles]:
                raise ValueError(f"Saved sample differs for {model}")
            frame = frame.astype(object)
        else:
            frame = pd.DataFrame(
                [
                    dict(
                        **p.to_dict(),
                        **dict.fromkeys(TOOL_COUNT_FIELDS, 0),
                        model=model,
                        use_tools=cfg.use_tools,
                        complete=False,
                        correct=False,
                        model_response="",
                        error="",
                        request_id="",
                    )
                    for p in puzzles
                ]
            ).astype(object)
        if "request_id" not in frame:
            frame["request_id"] = ""
        frames[model], files[model] = frame, path
        for index, puzzle in enumerate(puzzles):
            if str(frame.at[index, "complete"]).lower() != "true":
                pending.append((model, index, puzzle, folder / "cache"))
    write_json(manifest, settings)
    for profile_path, spec in profiles:
        write_json(profile_path, spec)
    for model, path in files.items():
        write_csv(frames[model], path)
    aggregate_results(exp_dir, cfg.protocol)
    semaphore = asyncio.Semaphore(cfg.num_threads)

    async def evaluate(model, index, puzzle, cache):
        async with semaphore:
            result = await PROTOCOLS[cfg.protocol](
                api,
                model,
                puzzle,
                cache,
                cfg.max_tokens,
                cfg.max_tool_rounds,
                cfg.temperature,
                use_tools=cfg.use_tools,
            )
            # No await during writes: tasks cannot interleave these checkpoints.
            for key, value in result.items():
                if key not in frames[model]:
                    frames[model][key] = pd.Series([None] * len(puzzles), dtype=object)
                frames[model].at[index, key] = value
            write_csv(frames[model], files[model])
            aggregate_results(exp_dir, cfg.protocol)
            LOGGER.info(
                "%s puzzle %s: complete=%s correct=%s",
                model,
                puzzle.puzzleId,
                result["complete"],
                result["correct"],
            )
            return result

    results = await asyncio.gather(*(evaluate(*item) for item in pending))
    failures = sum(not r["complete"] for r in results)
    if failures:
        raise RuntimeError(
            f"{failures} puzzles failed; details saved in puzzles.csv. Rerun the same command to resume."
        )
    return frames


async def run_proposer_judge(cfg, api):
    from omegaconf import OmegaConf

    from core.chess.judge import load_transcripts, run_judging

    pairs = []
    for pair in cfg.models:
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("proposer_judge models must be [proposer, judge] pairs")
        proposer, judge = (resolve_model(name) for name in pair)
        if any(s["endpoint"] == "completions" for s in (proposer, judge)):
            raise ValueError("Chess requires Chat, Responses, or Messages models")
        pairs.append((proposer["id"], judge["id"]))
    if not pairs:
        raise ValueError("Specify at least one [proposer, judge] pair")
    proposer_cfg = OmegaConf.merge(
        cfg,
        dict(
            protocol="direct_qa",
            models=list(dict.fromkeys(p for p, _ in pairs)),
        ),
    )
    await run_experiment(proposer_cfg, api)
    judge_cfg = OmegaConf.merge(
        cfg,
        dict(
            protocol="judge",
            models=list(dict.fromkeys(j for _, j in pairs)),
            exp_dir=str(Path(cfg.exp_dir) / "judgements"),
        ),
    )
    return await run_judging(judge_cfg, api, load_transcripts(cfg.exp_dir), set(pairs))


async def async_main(cfg):
    setup_environment(
        logger_level=cfg.logging,
        anthropic_tag=cfg.anthropic_tag,
        openai_tag=cfg.openai_tag,
    )
    api = ModelAPI(
        anthropic_num_threads=cfg.anthropic_num_threads,
        openai_fraction_rate_limit=cfg.openai_fraction_rate_limit,
        openrouter_timeout=cfg.openrouter_timeout,
        print_prompt_and_response=cfg.print_prompt_and_response,
        organization=cfg.organization,
    )
    try:
        return await run_experiment(cfg, api)
    finally:
        await api.close()


@hydra.main(version_base=None, config_path="../config", config_name="chess")
def main(cfg: DictConfig):
    asyncio.run(async_main(cfg))


if __name__ == "__main__":
    main()
