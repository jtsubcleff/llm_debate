"""Rescore saved chess proposals or judgements without model requests."""

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pandas as pd

from core.chess.protocols.direct_qa import score_response
from core.chess.protocols.judge import score_judgement
from core.chess.results import aggregate_results, write_csv


def rescore_experiment(exp_dir, write=False, protocol="direct_qa"):
    exp_dir = Path(exp_dir)
    paths = sorted((exp_dir / protocol).glob("*/puzzles.csv"))
    if not paths:
        raise ValueError(f"No {protocol} puzzle results found in {exp_dir}")
    updates, reports = [], []
    for path in paths:
        frame = pd.read_csv(
            path, keep_default_na=False, dtype={"puzzleId": str}
        ).astype(object)
        changed = gained = lost = skipped = rescored = 0
        for index, row in frame.iterrows():
            if str(row["complete"]).lower() != "true":
                continue
            trace = json.loads(row.get("tool_trace") or "[]")
            if protocol in ("debate", "consultancy"):
                trace = [
                    e
                    for e in trace
                    if e.get("round") == 1 and e.get("role") == "proposer"
                ]
            responses = [entry for entry in trace if "completion" in entry]
            # Use the final turn, not concatenated text from earlier tool rounds.
            # Without a saved stop reason we cannot safely rule out truncation.
            if not responses or responses[-1].get("tool_calls"):
                skipped += 1
                continue
            response = responses[-1]
            if "stop_reason" not in response:
                skipped += 1
                continue
            answer_column = "judgement" if protocol == "judge" else "final_move"
            if protocol == "judge":
                label = str(row["proposer_correct"]).lower()
                if label not in ("true", "false"):
                    raise ValueError("Invalid proposer_correct label")
                answer, correct = score_judgement(
                    response["completion"], label == "true"
                )
            else:
                puzzle = SimpleNamespace(
                    FEN=row["FEN"], expected_move=row["expected_move"]
                )
                answer, correct = score_response(puzzle, response["completion"])
            if protocol in ("debate", "consultancy"):
                from core.chess.protocols.interaction import score_proposal

                answer, correct = score_proposal(puzzle, trace)
            correct = bool(
                correct
                and response["stop_reason"]
                in ("STOP_SEQUENCE", "stop", "end_turn", "completed")
            )
            previous = str(row["correct"]).lower() == "true"
            gained += correct and not previous
            lost += previous and not correct
            changed += answer != row[answer_column] or correct != previous
            rescored += 1
            frame.at[index, answer_column] = answer
            frame.at[index, "correct"] = correct
        reports.append(
            dict(
                model=frame.model.iloc[0],
                rescored=rescored,
                changed=changed,
                gained=gained,
                lost=lost,
                skipped=skipped,
            )
        )
        if changed:
            updates.append((path, frame))
    if write and updates:
        suffix = f".pre-rescore-{uuid4().hex[:8]}.bak"
        # Back up every affected CSV before replacing any of them.
        for path in [p for p, _ in updates] + [exp_dir / "results.csv"]:
            if path.exists():
                shutil.copy2(path, path.with_name(path.name + suffix))
        for path, frame in updates:
            write_csv(frame, path)
        aggregate_results(exp_dir, protocol)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument(
        "--protocol",
        choices=("direct_qa", "debate", "consultancy", "judge"),
        default="direct_qa",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Back up and update puzzle CSVs and results.csv",
    )
    args = parser.parse_args()
    reports = rescore_experiment(args.exp_dir, write=args.write, protocol=args.protocol)
    for row in reports:
        print(
            f"{row['model']}: {row['rescored']} rescored, {row['changed']} changed, "
            f"{row['gained']} newly correct, {row['lost']} newly incorrect, "
            f"{row['skipped']} missing/unsupported traces"
        )
    print(
        "Results updated where changed; backups use .pre-rescore-*.bak."
        if args.write
        else "Preview only; use --write to save. Stop any run writing this directory first."
    )


if __name__ == "__main__":
    main()
