"""Incremental atomic CSV storage and task-specific scoring summaries."""

import json
from pathlib import Path

import pandas as pd

from core.load.chess import BUCKETS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def aggregate_results(exp_dir, protocol="direct_qa"):
    rows = []
    frames = []
    for path in sorted((Path(exp_dir) / protocol).glob("*/puzzles.csv")):
        df = pd.read_csv(
            path,
            keep_default_na=False,
            usecols=lambda name: name
            in {"model", "questioner", "complete", "correct", "rating_bucket"},
            dtype={"model": str},
        )
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    grouping = ["model", "questioner"] if protocol == "consultancy" else "model"
    for model, df in combined.groupby(grouping, sort=True) if frames else []:
        done = df[df.complete.astype(str).str.lower() == "true"]
        row = dict(
            model=model[0] if protocol == "consultancy" else model,
            protocol=protocol,
            planned=len(df),
            completed=len(done),
        )
        if protocol == "consultancy":
            row["questioner"] = model[1]
        for bucket in (*BUCKETS, "all"):
            subset = done if bucket == "all" else done[done.rating_bucket == bucket]
            successes = subset.correct.astype(str).str.lower().eq("true").sum()
            row[bucket] = successes / len(subset) if len(subset) else None
            row[f"{bucket}_count"] = len(subset)
        if protocol == "judge":
            row["accuracy"] = 100 * row["all"] if row["all"] is not None else None
        rows.append(row)
    write_csv(pd.DataFrame(rows), Path(exp_dir) / "results.csv")


def print_run_cost(exp_dir, protocol="direct_qa"):
    """Report recorded usage/cost once, including prior checkpoints on resume."""
    exp_dir = Path(exp_dir)
    roots = (
        [exp_dir / "direct_qa", exp_dir / "judgements" / "judge"]
        if protocol == "proposer_judge"
        else [exp_dir / protocol]
    )
    frames = [
        pd.read_csv(path, keep_default_na=False)
        for root in roots
        for path in sorted(root.glob("*/puzzles.csv"))
    ]
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    for field in ("cost", "input_tokens", "output_tokens", "api_calls"):
        values = (
            frame[field]
            if field in frame
            else pd.Series(index=frame.index, dtype=float)
        )
        frame[field] = pd.to_numeric(values, errors="coerce")
    # Unattempted rows have no usage or cost and should not count as unknown.
    attempted = (
        frame.complete.astype(str).str.lower().eq("true")
        | frame.cost.notna()
        | frame[["input_tokens", "output_tokens", "api_calls"]]
        .fillna(0)
        .gt(0)
        .any(axis=1)
    )
    frame = frame[attempted]
    print(f"\nEstimated cost of saved run ({protocol}):")
    print(
        "Includes prior checkpoints and cached turns once; not additional spending this invocation."
    )
    for model, rows in frame.groupby("model", sort=True):
        unknown = rows.cost.isna().sum()
        suffix = f" + unknown costs for {unknown} puzzle(s)" if unknown else ""
        print(
            f"  {model}: ${rows.cost.sum():,.6f}{suffix} | "
            f"{int(rows.input_tokens.sum()):,} input / "
            f"{int(rows.output_tokens.sum()):,} output tokens"
        )
    unknown = frame.cost.isna().sum()
    suffix = f" + unknown costs for {unknown} puzzle(s)" if unknown else ""
    print(f"Total estimated cost (USD): ${frame.cost.sum():,.6f}{suffix}")
