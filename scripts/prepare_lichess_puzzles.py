#!/usr/bin/env python3
"""Select rating-balanced Lichess puzzles and write a CSV sorted by rating.

Keep the first N puzzles in each bucket: <1300, 1300 <= rating <2300, and
2300+. N defaults to 1000. Stop reading once all buckets are full. If the
input ends first, write the available puzzles and report any shortfall.
Start at a random puzzle row, wrapping around to the beginning if needed.
The first run caches the row count in a .rows.json sidecar; subsequent runs
reuse it if the input is unchanged. Seeking streams up to the random start.
Use --seed to repeat the same selection.

No API calls or credentials are needed. Install dependencies with:

    pip install python-chess zstandard tqdm

Run from the repository root:

    python scripts/prepare_lichess_puzzles.py \
        --input chess_data/lichess_db_puzzle.csv.zst \
        --output chess_data/lichess_puzzles.csv \
        --puzzles-per-bucket 1000
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import tempfile
from pathlib import Path

from tqdm import tqdm

# Share the same schema, streaming decompression, and move validation with the
# date-splitting script. Importing these helpers performs no network requests.
if __package__:
    from .filter_lichess_puzzles import (
        OUTPUT_COLUMNS,
        normalized_position,
        randomized_source_rows,
    )
else:
    from filter_lichess_puzzles import (
        OUTPUT_COLUMNS,
        normalized_position,
        randomized_source_rows,
    )


BUCKET_LABELS = ("<1300", "1300-2299", "2300+")


def process(
    input_path: Path,
    output_path: Path,
    puzzles_per_bucket: int = 1000,
    seed: int | None = None,
) -> int:
    """Select up to the quota per bucket, transform moves, and sort on disk."""
    if puzzles_per_bucket <= 0:
        raise ValueError("puzzles_per_bucket must be a positive integer")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.TemporaryDirectory(prefix="lichess-puzzles-") as temp_dir:
        connection = sqlite3.connect(str(Path(temp_dir) / "puzzles.sqlite3"))
        try:
            connection.executescript(
                """
                CREATE TABLE puzzles (
                    sequence INTEGER PRIMARY KEY,
                    puzzle_id TEXT NOT NULL,
                    fen TEXT NOT NULL,
                    moves TEXT NOT NULL,
                    rating INTEGER NOT NULL,
                    themes TEXT NOT NULL,
                    game_url TEXT NOT NULL
                );
                CREATE INDEX puzzles_order ON puzzles(rating, puzzle_id, sequence);
                """
            )
            count = 0
            bucket_counts = [0, 0, 0]
            with randomized_source_rows(input_path, seed) as rows, tqdm(
                total=3 * puzzles_per_bucket,
                desc="Selecting puzzles",
                unit="puzzle",
                dynamic_ncols=True,
            ) as progress:
                for row_number, row in rows:
                    try:
                        rating = int(row["Rating"])
                        bucket = 0 if rating < 1300 else 1 if rating < 2300 else 2
                        if bucket_counts[bucket] >= puzzles_per_bucket:
                            continue
                        puzzle_id = row["PuzzleId"]
                        fen, moves = normalized_position(
                            row["FEN"], row["Moves"], puzzle_id
                        )
                        connection.execute(
                            """INSERT INTO puzzles(puzzle_id, fen, moves, rating, themes, game_url)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (
                                puzzle_id,
                                fen,
                                moves,
                                rating,
                                row["Themes"],
                                row["GameUrl"],
                            ),
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to process CSV row {row_number}: {exc}"
                        ) from exc
                    count += 1
                    bucket_counts[bucket] += 1
                    if count % 10000 == 0:
                        connection.commit()
                    progress.set_postfix(
                        {
                            label: f"{n}/{puzzles_per_bucket}"
                            for label, n in zip(BUCKET_LABELS, bucket_counts)
                        },
                        refresh=False,
                    )
                    progress.update(1)
                    if all(n == puzzles_per_bucket for n in bucket_counts):
                        break
            connection.commit()
            for label, selected in zip(BUCKET_LABELS, bucket_counts):
                if selected < puzzles_per_bucket:
                    tqdm.write(
                        f"Input exhausted: bucket {label} has {selected:,}/{puzzles_per_bucket:,} puzzles."
                    )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.writer(output)
                writer.writerow(OUTPUT_COLUMNS)
                rows = connection.execute(
                    """SELECT puzzle_id, fen, moves, rating, themes, game_url
                       FROM puzzles ORDER BY rating ASC, puzzle_id ASC, sequence ASC"""
                )
                with tqdm(
                    rows,
                    total=count,
                    desc=f"Writing {output_path.name}",
                    unit="puzzle",
                    dynamic_ncols=True,
                ) as progress:
                    for row in progress:
                        writer.writerow(row)
            return count
        finally:
            connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("chess_data/lichess_db_puzzle.csv.zst")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("chess_data/lichess_puzzles.csv")
    )
    parser.add_argument(
        "--puzzles-per-bucket",
        type=int,
        default=1000,
        help="Number to keep in each rating bucket (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible selection",
    )
    args = parser.parse_args()
    if args.puzzles_per_bucket <= 0:
        parser.error("--puzzles-per-bucket must be a positive integer")
    count = process(args.input, args.output, args.puzzles_per_bucket, args.seed)
    print(f"Wrote {count:,} puzzles to {args.output}")


if __name__ == "__main__":
    main()
