"""Matched four-condition chess experiments with shared proposals and prefixes."""

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from core.chess.interaction import run_interactions
from core.chess.judge import load_transcripts, run_judging
from core.chess.protocols.interaction import interaction_prefix, turn_schedule
from core.chess.results import write_csv, write_json
from core.chess.tools import CHESS_TOOLS
from core.chess.word_limits import PUBLIC_WORD_LIMITS
from core.llm_api.registry import resolve_model
from core.load.chess import load_puzzles, sample_puzzles
from core.prompts import load_template

CONDITIONS = (
    "single_consultancy",
    "multi_consultancy",
    "single_debate",
    "multi_debate",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def implementation_hashes():
    core = Path(__file__).resolve().parents[1]
    paths = [
        *(core / "chess").rglob("*.py"),
        *(core / "llm_api").glob("*.py"),
        core / "load/chess.py",
        core / "prompts.py",
    ]
    return {
        str(p.relative_to(core)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def runtime_versions():
    packages = {}
    for name in (
        "chess",
        "pandas",
        "numpy",
        "attrs",
        "pydantic",
        "hydra-core",
        "omegaconf",
        "openai",
        "anthropic",
        "PyYAML",
        "tiktoken",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return dict(python=platform.python_version(), packages=packages)


def suite_pairs(models):
    pairs = {}
    for pair in models:
        if isinstance(pair, str) or len(pair) != 2:
            raise ValueError("research_suite models must be [proposer, judge] pairs")
        proposer, judge = (resolve_model(name) for name in pair)
        if any(spec["endpoint"] == "completions" for spec in (proposer, judge)):
            raise ValueError("Chess requires Chat, Responses, or Messages models")
        pairs[(proposer["id"], judge["id"])] = (proposer, judge)
    if not pairs:
        raise ValueError("Specify at least one [proposer, judge] pair")
    return pairs


def saved_rows(frame):
    """Normalize pandas scalars/missing values to portable JSON-compatible data."""
    records = json.loads(frame.to_json(orient="records", double_precision=15))
    for row in records:
        # CSV round-trips turn optional nulls into empty strings. Their meaning
        # must not change source hashes or prevent a cached run from resuming.
        for key in ("cost", "request_id"):
            if row.get(key) == "":
                row[key] = None
        # Partially checkpointed columns can contain both booleans and blanks;
        # pandas then reads completed values as strings instead of bools.
        for key in (
            "complete",
            "correct",
            "proposer_correct",
            "proposal_valid",
            "interaction_valid",
            "verdict_valid",
        ):
            value = row.get(key)
            if isinstance(value, str) and value.lower() in ("true", "false"):
                row[key] = value.lower() == "true"
    return records


async def run_suite(cfg, api):
    from core.chess.run import model_directory

    turn_schedule("consultancy", cfg.num_rounds)
    conditions = list(cfg.suite_conditions)
    if (
        not conditions
        or len(set(conditions)) != len(conditions)
        or set(conditions) - set(CONDITIONS)
    ):
        raise ValueError(
            f"suite_conditions must select unique entries from {CONDITIONS}"
        )
    if any(c.startswith("multi_") for c in conditions) and cfg.num_rounds < 2:
        raise ValueError("Multi-round suite conditions require num_rounds >= 2")
    if cfg.max_tokens <= 0 or cfg.max_tool_rounds < 0 or cfg.num_threads <= 0:
        raise ValueError(
            "Require max_tokens > 0, max_tool_rounds >= 0, num_threads > 0"
        )
    pairs = suite_pairs(cfg.models)
    puzzles = sample_puzzles(load_puzzles(Path(cfg.dataset_path)), cfg.limit, cfg.seed)
    root = Path(cfg.exp_dir)
    manifest = dict(
        schema_version=1,
        word_limits=PUBLIC_WORD_LIMITS,
        protocol="research_suite",
        conditions=conditions,
        num_rounds=cfg.num_rounds,
        seed=cfg.seed,
        puzzles=[p.to_dict() for p in puzzles],
        models=[dict(proposer=p, judge=j) for p, j in pairs.values()],
        max_tokens=cfg.max_tokens,
        max_tool_rounds=cfg.max_tool_rounds,
        temperature=cfg.temperature,
        use_tools=cfg.use_tools,
        prompts={
            name: load_template(name) for name in ("chess_interaction", "chess_judge")
        },
        tools=CHESS_TOOLS,
        implementation=implementation_hashes(),
        runtime=runtime_versions(),
        transcript_policy="final_public_argument_anonymous_v2",
        judge_use_tools=True,
    )
    path = root / "chess_suite.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(
            "Suite inputs, profiles, prompts, or implementation changed; choose a new exp_dir"
        )
    write_json(path, manifest)

    def stage(protocol, folder, models, rounds):
        return OmegaConf.merge(
            cfg,
            dict(
                protocol=protocol, exp_dir=str(folder), models=models, num_rounds=rounds
            ),
        )

    try:
        # One mode-neutral opening per proposer/puzzle, independent of judge and
        # condition. Every branch consumes this exact cached answer and argument.
        # Consultancy requires two resolvable model slots, so [p, p] supplies a
        # placeholder questioner; one round calls only the proposer. [p, None]
        # would fail model resolution even though the questioner is never used.
        proposers = list(dict.fromkeys(p for p, _ in pairs))
        frames = await run_interactions(
            stage(
                "consultancy", root / "_work/proposals", [[p, p] for p in proposers], 1
            ),
            api,
            puzzles,
            prompt_template=manifest["prompts"]["chess_interaction"],
        )
        openings = {
            (key[0], row["puzzleId"]): row
            for key, frame in frames.items()
            for row in saved_rows(frame)
        }
        generated = {}
        for mode in ("debate", "consultancy"):
            selected = [c for c in conditions if c.endswith(mode)]
            if not selected:
                continue
            rounds = cfg.num_rounds if "multi_" + mode in selected else 1
            if mode == "consultancy" and rounds == 1:
                generated[mode] = {
                    (p, j): {
                        puzzle.puzzleId: openings[p, puzzle.puzzleId]
                        for puzzle in puzzles
                    }
                    for p, j in pairs
                }
                continue
            models = proposers if mode == "debate" else [list(pair) for pair in pairs]
            frames = await run_interactions(
                stage(mode, root / "_work" / mode, models, rounds),
                api,
                puzzles,
                openings,
                prompt_template=manifest["prompts"]["chess_interaction"],
            )
            generated[mode] = {
                key: {r["puzzleId"]: r for r in saved_rows(frame)}
                for key, frame in frames.items()
            }

        for condition in conditions:
            mode = condition.split("_", 1)[1]
            rounds = 1 if condition.startswith("single_") else cfg.num_rounds
            folder = root / "conditions" / condition
            for proposer, judge in pairs:
                key = (proposer, proposer) if mode == "debate" else (proposer, judge)
                rows = []
                for puzzle in puzzles:
                    original = generated[mode][key][puzzle.puzzleId]
                    result = interaction_prefix(original, puzzle, mode, rounds)
                    rows.append(
                        dict(
                            **puzzle.to_dict(),
                            **result,
                            model=proposer,
                            questioner=judge
                            if mode == "consultancy" and rounds > 1
                            else "",
                            target_judge=judge,
                            condition=condition,
                            proposal_id=digest(openings[proposer, puzzle.puzzleId]),
                        )
                    )
                name = model_directory(proposer) + "__" + model_directory(judge)
                write_csv(pd.DataFrame(rows), folder / mode / name / "puzzles.csv")
            # Explicit targets avoid a Cartesian product, and ensure the final
            # judge is exactly the questioner profile in multi consultancy.
            sources = [
                s
                for s in load_transcripts(folder, allow_empty=True)
                if s.get("proposal_valid", True)
            ]
            if sources:
                judge_cfg = stage(
                    "judge",
                    folder / "judgements",
                    list(dict.fromkeys(j for _, j in pairs)),
                    rounds,
                )
                await run_judging(
                    judge_cfg,
                    api,
                    sources,
                    prompt_template=manifest["prompts"]["chess_judge"],
                )
    finally:
        summarize_suite(root, manifest)
    return pd.read_csv(root / "results.csv")


def summarize_suite(root, manifest):
    """Keep planned denominators and paired comparisons visible after failures."""
    root = Path(root)
    evaluations = []
    for condition in manifest["conditions"]:
        mode = condition.split("_", 1)[1]
        folder = root / "conditions" / condition
        proposals = {}
        for path in (folder / mode).glob("*/puzzles.csv"):
            for row in saved_rows(
                pd.read_csv(path, keep_default_na=False, dtype={"puzzleId": str})
            ):
                proposals[(row["model"], row["target_judge"], row["puzzleId"])] = row
        judged = {}
        for path in (folder / "judgements/judge").glob("*/puzzles.csv"):
            for row in saved_rows(
                pd.read_csv(path, keep_default_na=False, dtype={"puzzleId": str})
            ):
                judged[(row["proposer"], row["model"], row["puzzleId"])] = row
        for pair in manifest["models"]:
            proposer, judge = pair["proposer"]["id"], pair["judge"]["id"]
            for puzzle in manifest["puzzles"]:
                key = (proposer, judge, puzzle["puzzleId"])
                proposal, verdict = proposals.get(key, {}), judged.get(key, {})
                done = str(verdict.get("complete", False)).lower() == "true"
                evaluations.append(
                    dict(
                        condition=condition,
                        proposer=proposer,
                        judge=judge,
                        puzzleId=puzzle["puzzleId"],
                        rating_bucket=puzzle["rating_bucket"],
                        proposal_id=proposal.get("proposal_id", ""),
                        proposed_move=proposal.get("final_move", ""),
                        proposer_correct=proposal.get("correct", None),
                        proposal_valid=proposal.get("proposal_valid", False),
                        interaction_valid=proposal.get("interaction_valid", False),
                        generated=bool(proposal),
                        complete=done,
                        judgement=verdict.get("judgement", ""),
                        verdict_valid=verdict.get("verdict_valid", False),
                        correct=bool(
                            done
                            and str(verdict.get("correct", False)).lower() == "true"
                        ),
                    )
                )
    frame = pd.DataFrame(evaluations)
    write_csv(frame, root / "evaluations.csv")
    rows = []
    # The same puzzle IDs must be judged in all selected conditions before they
    # enter matched accuracy; overall coverage remains explicit as a separate metric.
    for (proposer, judge), pair in frame.groupby(["proposer", "judge"], sort=True):
        matched_ids = (
            set.intersection(
                *[
                    set(g.loc[g.complete, "puzzleId"])
                    for _, g in pair.groupby("condition")
                ]
            )
            if len(pair.condition.unique()) == len(manifest["conditions"])
            else set()
        )
        for condition, group in pair.groupby("condition", sort=True):
            done = group[group.complete]
            matched = done[done.puzzleId.isin(matched_ids)]
            truth = done.proposer_correct.astype(str).str.lower().eq("true")
            yes = done.judgement.eq("Yes") & done.verdict_valid.eq(True)
            no = done.judgement.eq("No") & done.verdict_valid.eq(True)
            tp, tn, fp, fn = (
                int((truth & yes).sum()),
                int((~truth & no).sum()),
                int((~truth & yes).sum()),
                int((truth & no).sum()),
            )
            positives, negatives = int(truth.sum()), int((~truth).sum())
            rows.append(
                dict(
                    condition=condition,
                    proposer=proposer,
                    judge=judge,
                    planned=len(group),
                    completed=len(done),
                    unjudged=len(group) - len(done),
                    missing_transcripts=int((~group.generated).sum()),
                    invalid_proposals=int(
                        (group.generated & ~group.proposal_valid.eq(True)).sum()
                    ),
                    invalid_interactions=int(
                        (group.generated & ~group.interaction_valid.eq(True)).sum()
                    ),
                    invalid_verdicts=int((~done.verdict_valid.eq(True)).sum()),
                    accuracy=100 * done.correct.mean() if len(done) else None,
                    matched_count=len(matched),
                    matched_accuracy=100 * matched.correct.mean()
                    if len(matched)
                    else None,
                    positives=positives,
                    negatives=negatives,
                    tp=tp,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    balanced_accuracy=50 * (tp / positives + tn / negatives)
                    if positives and negatives
                    else None,
                )
            )
    write_csv(pd.DataFrame(rows), root / "results.csv")
    # Count each functional response cache exactly once. Derived prefixes and
    # reused openings have logical costs but must not inflate actual suite cost.
    usage = []
    for path in root.rglob("cache/**/*.json"):
        value = json.loads(path.read_text())
        if "completion" in value and "model_id" in value:
            usage.append(
                dict(
                    model=value["model_id"],
                    cost=value.get("cost"),
                    input_tokens=value.get("input_tokens") or 0,
                    output_tokens=value.get("output_tokens") or 0,
                )
            )
    costs = []
    if usage:
        for model, group in pd.DataFrame(usage).groupby("model", sort=True):
            costs.append(
                dict(
                    model=model,
                    cost=group.cost.sum(),
                    unknown_cost_calls=int(group.cost.isna().sum()),
                    input_tokens=int(group.input_tokens.sum()),
                    output_tokens=int(group.output_tokens.sum()),
                    api_calls=len(group),
                )
            )
    write_csv(
        pd.DataFrame(
            costs,
            columns=[
                "model",
                "cost",
                "unknown_cost_calls",
                "input_tokens",
                "output_tokens",
                "api_calls",
            ],
        ),
        root / "costs.csv",
    )
