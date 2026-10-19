#!/usr/bin/env python3
"""Replace story text in a debate results CSV with a short placeholder."""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def replace_story_fields(value: Any, placeholder: str) -> Any:
    """Recursively replace JSON fields named ``story``."""
    if isinstance(value, dict):
        return {
            key: placeholder
            if key.lower() == "story"
            else replace_story_fields(child, placeholder)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [replace_story_fields(child, placeholder) for child in value]
    return value


def redact_csv(input_path: Path, output_path: Path, placeholder: str) -> int:
    """Write a redacted copy and return the number of rows processed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    row_count = 0
    try:
        with input_path.open("r", newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {input_path}")

            with tempfile.NamedTemporaryFile(
                "w",
                newline="",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as target:
                temp_path = Path(target.name)
                writer = csv.DictWriter(
                    target,
                    fieldnames=reader.fieldnames,
                    extrasaction="ignore",
                )
                writer.writeheader()

                for row_count, row in enumerate(reader, start=1):
                    if "story" in row and row["story"]:
                        row["story"] = placeholder

                    transcript = row.get("transcript")
                    if transcript:
                        try:
                            parsed = json.loads(transcript)
                        except json.JSONDecodeError as error:
                            raise ValueError(
                                f"Invalid transcript JSON on CSV row {row_count + 1}"
                            ) from error
                        parsed = replace_story_fields(parsed, placeholder)
                        row["transcript"] = json.dumps(
                            parsed, ensure_ascii=False, separators=(",", ":")
                        )

                    writer.writerow(row)

        os.replace(temp_path, output_path)
        return row_count
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, help="Input debate results CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output CSV (default: <input>_story_redacted.csv)",
    )
    parser.add_argument(
        "--placeholder", default="[STORY OMITTED]", help="Replacement text"
    )
    args = parser.parse_args()

    input_path = args.results_csv
    output_path = args.output or input_path.with_name(
        f"{input_path.stem}_story_redacted{input_path.suffix or '.csv'}"
    )
    rows = redact_csv(input_path, output_path, args.placeholder)
    print(f"Redacted story fields in {rows} rows: {output_path}")


if __name__ == "__main__":
    main()
