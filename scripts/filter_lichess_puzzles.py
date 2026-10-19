#!/usr/bin/env python3
"""Split the Lichess puzzle database by source-game date and normalize moves.

The source GameUrl links to a Lichess game but does not encode its date, so the
date is read from Lichess's game export API. A temporary SQLite database
caches game dates and stores transformed rows so sorting does not require
keeping the full dataset in memory.

Install dependencies with ``pip install python-chess zstandard tqdm``. Example:
Set ``LICHESS_PAT`` in ``SECRETS`` before running (or pass another file with
``--secrets``).

    python scripts/filter_lichess_puzzles.py \
        --input chess_data/lichess_db_puzzle.csv.zst \
        --before-output chess_data/lichess_puzzles_before_2024-06-01.csv \
        --after-output chess_data/lichess_puzzles_on_or_after_2024-06-01.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sqlite3
import tempfile
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import chess
import zstandard
from tqdm import tqdm

INPUT_COLUMNS = [
    "PuzzleId",
    "FEN",
    "Moves",
    "Rating",
    "RatingDeviation",
    "Popularity",
    "NbPlays",
    "Themes",
    "GameUrl",
    "OpeningTags",
    "DailyDate",
]
OUTPUT_COLUMNS = ["PuzzleId", "FEN", "Moves", "Rating", "Themes", "GameUrl"]
CUTOFF = date(2024, 6, 1)
DATE_HEADER = re.compile(r'^\[Date "(\d{4})\.(\d{2})\.(\d{2})"\]$', re.MULTILINE)


def game_id_from_url(game_url: str) -> str:
    """Return the Lichess game ID from a standard game URL."""
    parts = [part for part in urlparse(game_url).path.split("/") if part]
    if not parts or not re.fullmatch(r"[A-Za-z0-9_-]+", parts[0]):
        raise ValueError(f"Cannot extract a Lichess game ID from GameUrl {game_url!r}")
    return parts[0]


def load_lichess_token(secrets_path: Path) -> str:
    """Load LICHESS_PAT from the project's KEY=VALUE secrets file."""
    if not secrets_path.is_file():
        raise FileNotFoundError(f"Secrets file not found: {secrets_path}")
    for line in secrets_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "LICHESS_PAT":
            token = value.strip().strip("\"'")
            if token:
                return token
            break
    raise ValueError(f"LICHESS_PAT is missing or empty in {secrets_path}")


def fetch_game_date(game_id: str, token: str, retries: int = 5) -> date:
    """Fetch the PGN date from the authenticated Lichess game export API."""
    # Lichess's documented single-game export API is rooted at /game/export,
    # without the /api prefix used by many other endpoints.
    url = f"https://lichess.org/game/export/{game_id}"
    request = Request(
        url,
        headers={
            "Accept": "application/x-chess-pgn",
            "Authorization": f"Bearer {token}",
            "User-Agent": "lichess-puzzle-filter/1.0",
        },
    )
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=30) as response:
                pgn = response.read().decode("utf-8", errors="replace")
            match = DATE_HEADER.search(pgn)
            if match is None:
                raise ValueError(
                    f"Lichess PGN for game {game_id} has no usable [Date] header"
                )
            year, month, day = (int(part) for part in match.groups())
            return date(year, month, day)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(
                    f"Could not fetch game {game_id} (HTTP {exc.code})"
                ) from exc
            delay = 60 if exc.code == 429 else min(2**attempt, 30)
        except (URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not fetch game {game_id}: {exc}") from exc
            delay = min(2**attempt, 30)
        tqdm.write(
            f"Retrying game {game_id} in {delay}s (attempt {attempt + 2}/{retries})"
        )
        time.sleep(delay)
    raise AssertionError("retry loop exited unexpectedly")


def normalized_position(fen: str, moves_text: str, puzzle_id: str) -> tuple[str, str]:
    """Play the first UCI move, then verify the remaining sequence from there."""
    move_tokens = moves_text.split()
    if not move_tokens:
        raise ValueError(f"Puzzle {puzzle_id} has no moves")

    board = chess.Board(fen)
    try:
        board.push_uci(move_tokens[0])
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError) as exc:
        raise ValueError(
            f"Puzzle {puzzle_id} has an invalid first move {move_tokens[0]!r}"
        ) from exc

    new_fen = board.fen()
    remaining_moves = move_tokens[1:]
    for ply, move in enumerate(remaining_moves, start=2):
        try:
            board.push_uci(move)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError) as exc:
            raise AssertionError(
                f"Puzzle {puzzle_id} has illegal move {move!r} at ply {ply} from its updated FEN"
            ) from exc
    return new_fen, " ".join(remaining_moves)


def open_source_csv(path: Path) -> io.TextIOWrapper:
    compressed = path.open("rb")
    reader = zstandard.ZstdDecompressor().stream_reader(compressed)
    return io.TextIOWrapper(reader, encoding="utf-8", newline="")


def create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE game_dates (game_id TEXT PRIMARY KEY, played_on TEXT NOT NULL);
        CREATE TABLE puzzles (
            sequence INTEGER PRIMARY KEY,
            period TEXT NOT NULL,
            puzzle_id TEXT NOT NULL,
            fen TEXT NOT NULL,
            moves TEXT NOT NULL,
            rating INTEGER NOT NULL,
            themes TEXT NOT NULL,
            game_url TEXT NOT NULL
        );
        CREATE INDEX puzzles_order ON puzzles(period, rating, puzzle_id, sequence);
        """
    )


def process(
    input_path: Path, before_path: Path, after_path: Path, token: str
) -> tuple[int, int]:
    if before_path.resolve() == after_path.resolve():
        raise ValueError("The two output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with tempfile.NamedTemporaryFile(
        prefix="lichess-puzzles-", suffix=".sqlite3"
    ) as temp_db:
        connection = sqlite3.connect(temp_db.name)
        try:
            create_database(connection)
            before_count = after_count = 0
            insert_sql = """INSERT INTO puzzles
                (period, puzzle_id, fen, moves, rating, themes, game_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)"""

            with open_source_csv(input_path) as source, tqdm(
                desc="Processing puzzles", unit="puzzle", dynamic_ncols=True
            ) as progress:
                reader = csv.DictReader(source)
                if reader.fieldnames != INPUT_COLUMNS:
                    raise ValueError(
                        f"Unexpected CSV columns. Expected {INPUT_COLUMNS!r}, got {reader.fieldnames!r}"
                    )

                for row_number, row in enumerate(reader, start=2):
                    try:
                        puzzle_id = row["PuzzleId"]
                        game_id = game_id_from_url(row["GameUrl"])
                        cached = connection.execute(
                            "SELECT played_on FROM game_dates WHERE game_id = ?",
                            (game_id,),
                        ).fetchone()
                        if cached:
                            played_on = date.fromisoformat(cached[0])
                        else:
                            played_on = fetch_game_date(game_id, token)
                            connection.execute(
                                "INSERT INTO game_dates(game_id, played_on) VALUES (?, ?)",
                                (game_id, played_on.isoformat()),
                            )

                        new_fen, remaining_moves = normalized_position(
                            row["FEN"], row["Moves"], puzzle_id
                        )
                        rating = int(row["Rating"])
                        period = "before" if played_on < CUTOFF else "after"
                        connection.execute(
                            insert_sql,
                            (
                                period,
                                puzzle_id,
                                new_fen,
                                remaining_moves,
                                rating,
                                row["Themes"],
                                row["GameUrl"],
                            ),
                        )
                        if period == "before":
                            before_count += 1
                        else:
                            after_count += 1
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to process CSV row {row_number}: {exc}"
                        ) from exc

                    if row_number % 10000 == 0:
                        connection.commit()
                    progress.set_postfix(
                        before=before_count, on_or_after=after_count, refresh=False
                    )
                    progress.update(1)

            connection.commit()
            write_output(connection, "before", before_path, before_count)
            write_output(connection, "after", after_path, after_count)
            return before_count, after_count
        finally:
            connection.close()


def write_output(
    connection: sqlite3.Connection, period: str, path: Path, total: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(OUTPUT_COLUMNS)
        rows = connection.execute(
            """SELECT puzzle_id, fen, moves, rating, themes, game_url
               FROM puzzles WHERE period = ? ORDER BY rating ASC, puzzle_id ASC, sequence ASC""",
            (period,),
        )
        with tqdm(
            rows,
            total=total,
            desc=f"Writing {path.name}",
            unit="puzzle",
            dynamic_ncols=True,
        ) as progress:
            for row in progress:
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("lichess_db_puzzle.csv.zst"))
    parser.add_argument(
        "--before-output",
        type=Path,
        default=Path("lichess_puzzles_before_2024-06-01.csv"),
    )
    parser.add_argument(
        "--after-output",
        type=Path,
        default=Path("lichess_puzzles_on_or_after_2024-06-01.csv"),
    )
    parser.add_argument(
        "--secrets", type=Path, default=Path("SECRETS"), help="KEY=VALUE secrets file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = load_lichess_token(args.secrets)
    before_count, after_count = process(
        args.input, args.before_output, args.after_output, token
    )
    print(f"Wrote {before_count:,} puzzles before {CUTOFF} to {args.before_output}")
    print(f"Wrote {after_count:,} puzzles on or after {CUTOFF} to {args.after_output}")


if __name__ == "__main__":
    main()
