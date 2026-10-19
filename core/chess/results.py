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
    for path in sorted((Path(exp_dir) / protocol).glob("*/puzzles.csv")):
        df = pd.read_csv(
            path,
            keep_default_na=False,
            usecols=["model", "complete", "correct", "rating_bucket"],
            dtype={"model": str},
        )
        done = df[df.complete.astype(str).str.lower() == "true"]
        row = dict(
            model=df.model.iloc[0],
            protocol=protocol,
            planned=len(df),
            completed=len(done),
        )
        for bucket in (*BUCKETS, "all"):
            subset = done if bucket == "all" else done[done.rating_bucket == bucket]
            successes = subset.correct.astype(str).str.lower().eq("true").sum()
            row[bucket] = successes / len(subset) if len(subset) else None
            row[f"{bucket}_count"] = len(subset)
        rows.append(row)
    write_csv(pd.DataFrame(rows), Path(exp_dir) / "results.csv")
