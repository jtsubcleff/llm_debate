"""Prepared Lichess puzzles, shared by chess experiment protocols.

The CSV FEN is already the solver's position; do not play a setup move.
"""

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import chess

BUCKETS = ("Low", "Medium", "High")
PUZZLES_CSV_PATH = "chess_data/lichess_post_2024-06-01.csv"


def rating_bucket(rating: int) -> str:
    return "Low" if rating < 1300 else "Medium" if rating <= 2300 else "High"


@dataclass(frozen=True)
class ChessPuzzle:
    puzzleId: str
    FEN: str
    moves: str
    rating: int
    rating_bucket: str

    @property
    def expected_move(self):
        board = chess.Board(self.FEN)
        return board.san(board.parse_uci(self.moves.split()[0]))

    def to_dict(self):
        return asdict(self)


def load_puzzles(path=Path(PUZZLES_CSV_PATH)):
    buckets = {bucket: [] for bucket in BUCKETS}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"PuzzleId", "FEN", "Moves", "Rating"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Chess CSV requires columns: {sorted(required)}")
        for row in reader:
            puzzle_id = row["PuzzleId"]
            try:
                if not puzzle_id or puzzle_id in seen:
                    raise ValueError("Empty or duplicate puzzle ID")
                seen.add(puzzle_id)
                rating = int(row["Rating"])
                board = chess.Board(row["FEN"])
                if not board.is_valid() or not row["Moves"].split():
                    raise ValueError("Invalid board or empty solution")
                for move in row["Moves"].split():
                    board.push_uci(move)
                bucket = rating_bucket(rating)
                buckets[bucket].append(
                    ChessPuzzle(puzzle_id, row["FEN"], row["Moves"], rating, bucket)
                )
            except ValueError as exc:
                raise ValueError(f"Invalid puzzle {puzzle_id!r}: {exc}") from exc
    return buckets


def sample_puzzles(buckets, limit: int, seed: int = 0):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0 or limit % 3:
        raise ValueError("limit must be a positive integer divisible by 3")
    per_bucket = limit // 3
    rng = random.Random(seed)
    selected = []
    for bucket in BUCKETS:
        if len(buckets[bucket]) < per_bucket:
            raise ValueError(
                f"{bucket} has {len(buckets[bucket])} puzzles; need {per_bucket}"
            )
        selected.extend(rng.sample(buckets[bucket], per_bucket))
    rng.shuffle(selected)
    return selected
