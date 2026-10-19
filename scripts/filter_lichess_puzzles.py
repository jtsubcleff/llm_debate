#!/usr/bin/env python3
"""Split the Lichess puzzle database by source-game date and normalize moves.

The source GameUrl links to a Lichess game but does not encode its date, so the
date is read from Lichess's game export API in batches of up to 300 games.
Only PGN headers are downloaded. Dates are saved after each batch in
lichess_game_dates.sqlite3 beside the input (override with --date-cache).
A temporary SQLite database stores transformed rows for sorting on disk.

Keep the first 1000 puzzles per rating bucket (<1300, 1300-2299, 2300+) in
each date split by default. Set --puzzles-per-bucket to change this quota.
Stop reading once all six quotas are filled; report shortfalls at end of input.
Start at a uniformly random puzzle row and wrap around if needed, visiting
each row at most once. The first run counts CSV rows; subsequent runs reuse
a small .rows.json sidecar if the input's path, size, and modification time
match. Seeking still requires decompression up to the start. Use --seed for
reproducible selection, or --batch-size to change the API batch size.

Install dependencies with ``pip install python-chess zstandard tqdm``. Example:
Set ``LICHESS_PAT`` in ``SECRETS`` before running (or pass another file with
``--secrets``).

    python scripts/filter_lichess_puzzles.py \
        --input chess_data/lichess_db_puzzle.csv.zst \
        --before-output chess_data/lichess_pre_2024-06-01.csv \
        --after-output chess_data/lichess_post_2024-06-01.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import re
import sqlite3
import tempfile
import time
from contextlib import closing, contextmanager
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


def download_pgn(request: Request, retries: int = 5) -> str:
    """Make one request at a time, respecting Lichess's rate-limit cooldown."""
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                raise RuntimeError(f"Lichess export failed (HTTP {exc.code})") from exc
            delay = 60 if exc.code == 429 else min(2**attempt, 30)
        except (URLError, TimeoutError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"Lichess export failed: {exc}") from exc
            delay = min(2**attempt, 30)
        tqdm.write(
            f"Retrying Lichess export in {delay}s (attempt {attempt + 2}/{retries})"
        )
        time.sleep(delay)
    raise AssertionError("retry loop exited unexpectedly")


def fetch_game_date(game_id: str, token: str, retries: int = 5) -> date:
    """Fetch one game's original PGN date without downloading moves."""
    request = Request(
        f"https://lichess.org/game/export/{game_id}?moves=false&clocks=false&evals=false",
        headers={
            "Accept": "application/x-chess-pgn",
            "Authorization": f"Bearer {token}",
        },
    )
    match = DATE_HEADER.search(download_pgn(request, retries))
    if match is None:
        raise ValueError(f"Lichess PGN for game {game_id} has no usable [Date] header")
    return date(*(int(part) for part in match.groups()))


def fetch_game_dates(game_ids: list[str], token: str) -> dict[str, date]:
    """Export up to 300 games' headers, matching responses by ID, not order."""
    if not 1 <= len(game_ids) <= 300:
        raise ValueError("A date lookup batch must contain between 1 and 300 game IDs")
    request = Request(
        "https://lichess.org/api/games/export/_ids?moves=false&clocks=false&evals=false&opening=false",
        data=",".join(game_ids).encode("ascii"),
        headers={
            "Accept": "application/x-chess-pgn",
            "Content-Type": "text/plain",
            "Authorization": f"Bearer {token}",
            "User-Agent": "lichess-puzzle-filter/1.0",
        },
        method="POST",
    )
    # Use the PGN Date rather than upload timestamps, so imported games retain
    # the date on which they were played. Bulk exports arrive newest first.
    import chess.pgn

    result = {}
    requested = set(game_ids)
    source = io.StringIO(download_pgn(request))
    while (headers := chess.pgn.read_headers(source)) is not None:
        game_id = game_id_from_url(headers.get("Site", ""))
        if game_id in requested:
            try:
                result[game_id] = date.fromisoformat(headers["Date"].replace(".", "-"))
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Lichess PGN for game {game_id} has no usable [Date] header"
                ) from exc
    missing = requested - result.keys()
    if missing:
        raise RuntimeError(
            f"Lichess did not return dates for games: {', '.join(sorted(missing))}"
        )
    return result


def open_date_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    connection.execute(
        "CREATE TABLE IF NOT EXISTS game_dates (game_id TEXT PRIMARY KEY, played_on TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def dated_rows(rows, cache, token, bucket_counts, quota, batch_size, progress):
    """Look ahead in bounded batches without changing candidate order."""
    iterator = iter(rows)
    exhausted = False
    cache_hits = batches = 0
    while not exhausted:
        pending = []
        missing = set()
        remaining = sum(quota - n for counts in bucket_counts.values() for n in counts)
        limit = min(batch_size, remaining)
        if limit <= 0:
            return
        while len(pending) < limit:
            try:
                row_number, row = next(iterator)
            except StopIteration:
                exhausted = True
                break
            try:
                rating = int(row["Rating"])
                bucket = 0 if rating < 1300 else 1 if rating < 2300 else 2
                if all(counts[bucket] >= quota for counts in bucket_counts.values()):
                    continue
                game_id = game_id_from_url(row["GameUrl"])
                cached = cache.execute(
                    "SELECT played_on FROM game_dates WHERE game_id = ?", (game_id,)
                ).fetchone()
                played_on = date.fromisoformat(cached[0]) if cached else None
                if played_on is None:
                    missing.add(game_id)
                else:
                    cache_hits += 1
                pending.append((row_number, row, rating, bucket, game_id, played_on))
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to process CSV row {row_number}: {exc}"
                ) from exc
        dates = {}
        if missing:
            progress.set_description(f"Fetching {len(missing)} game dates")
            dates = fetch_game_dates(sorted(missing), token)
            cache.executemany(
                "INSERT OR REPLACE INTO game_dates(game_id, played_on) VALUES (?, ?)",
                (
                    (game_id, played_on.isoformat())
                    for game_id, played_on in dates.items()
                ),
            )
            # Save every successful batch even if a later lookup/validation fails.
            cache.commit()
            batches += 1
        progress.set_description(f"Selecting (batches={batches}, cached={cache_hits})")
        for row_number, row, rating, bucket, game_id, played_on in pending:
            yield row_number, row, rating, bucket, played_on or dates[game_id]


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


@contextmanager
def randomized_source_rows(path: Path, seed: int | None = None):
    """Yield (original CSV row number, row) from a random circular starting point.

    Ordinary zstd streams cannot be opened at arbitrary compressed offsets.
    Cache the CSV record count, then decompress up to the randomly selected row.
    Close both streams even when the caller stops early after filling quotas.
    """
    total = source_row_count(path)

    if total == 0:
        yield iter(())
        return
    start = random.Random(seed).randrange(total)
    tqdm.write(
        f"Starting at puzzle {start + 1:,} of {total:,} (CSV row {start + 2:,})."
    )

    def circular_rows():
        with open_source_csv(path) as source:
            # Skip with csv.reader: constructing millions of dictionaries here
            # contributes nothing because these rows are not being selected.
            raw_rows = (row for row in csv.reader(source) if row)
            next(raw_rows)  # Header was checked by source_row_count.
            with tqdm(
                range(start),
                desc="Seeking random start",
                unit="puzzle",
                dynamic_ncols=True,
            ) as progress:
                for _ in progress:
                    next(raw_rows)
            reader = csv.DictReader(source, fieldnames=INPUT_COLUMNS)
            for row_number, row in enumerate(reader, start=start + 2):
                yield row_number, row
        if start:
            tqdm.write("Reached end of input; continuing from the beginning.")
            with open_source_csv(path) as source:
                reader = csv.DictReader(source)
                for index in range(start):
                    yield index + 2, next(reader)

    with closing(circular_rows()) as rows:
        yield rows


def source_row_count(path: Path) -> int:
    """Cache only a row count; changed inputs are detected by size and mtime."""
    stat = path.stat()
    fingerprint = {
        "source": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    cache_path = path.with_name(path.name + ".rows.json")
    try:
        metadata = json.loads(cache_path.read_text(encoding="utf-8"))
        cached = metadata.get("rows")
        valid_cache = (
            all(metadata.get(key) == value for key, value in fingerprint.items())
            and type(cached) is int
            and cached >= 0
        )
    except (OSError, ValueError, AttributeError):
        valid_cache = False
    with open_source_csv(path) as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header != INPUT_COLUMNS:
            raise ValueError(
                f"Unexpected CSV columns. Expected {INPUT_COLUMNS!r}, got {header!r}"
            )
        if valid_cache:
            tqdm.write(f"Using cached puzzle count: {cached:,}.")
            return cached
        with tqdm(
            reader,
            desc="Counting puzzles (first run)",
            unit="puzzle",
            dynamic_ncols=True,
        ) as progress:
            total = sum(1 for row in progress if row)
    latest = path.stat()
    if (latest.st_size, latest.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise RuntimeError(
            "Input changed while counting puzzles; retry with a stable input file"
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump({**fingerprint, "rows": total}, file)
        os.replace(temporary, cache_path)
    except OSError as exc:
        tqdm.write(f"Could not save puzzle count cache: {exc}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return total


def create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
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
    input_path: Path,
    before_path: Path,
    after_path: Path,
    token: str,
    puzzles_per_bucket: int = 1000,
    seed: int | None = None,
    date_cache_path: Path | None = None,
    batch_size: int = 300,
) -> tuple[int, int]:
    if puzzles_per_bucket <= 0:
        raise ValueError("puzzles_per_bucket must be a positive integer")
    if before_path.resolve() == after_path.resolve():
        raise ValueError("The two output paths must be different")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not 1 <= batch_size <= 300:
        raise ValueError("batch_size must be between 1 and 300")
    date_cache_path = (
        date_cache_path or input_path.parent / "lichess_game_dates.sqlite3"
    )
    paths = [
        input_path.resolve(),
        before_path.resolve(),
        after_path.resolve(),
        date_cache_path.resolve(),
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("Input, output, and date-cache paths must all be different")

    with tempfile.NamedTemporaryFile(
        prefix="lichess-puzzles-", suffix=".sqlite3"
    ) as temp_db, closing(open_date_cache(date_cache_path)) as date_cache:
        connection = sqlite3.connect(temp_db.name)
        try:
            create_database(connection)
            before_count = after_count = 0
            bucket_labels = ("<1300", "1300-2299", "2300+")
            bucket_counts = {"before": [0, 0, 0], "after": [0, 0, 0]}
            insert_sql = """INSERT INTO puzzles
                (period, puzzle_id, fen, moves, rating, themes, game_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)"""

            with randomized_source_rows(input_path, seed) as rows, tqdm(
                total=6 * puzzles_per_bucket,
                desc="Selecting puzzles",
                unit="puzzle",
                dynamic_ncols=True,
            ) as progress:
                for row_number, row, rating, bucket, played_on in dated_rows(
                    rows,
                    date_cache,
                    token,
                    bucket_counts,
                    puzzles_per_bucket,
                    batch_size,
                    progress,
                ):
                    try:
                        puzzle_id = row["PuzzleId"]
                        period = "before" if played_on < CUTOFF else "after"
                        if bucket_counts[period][bucket] >= puzzles_per_bucket:
                            continue
                        new_fen, remaining_moves = normalized_position(
                            row["FEN"], row["Moves"], puzzle_id
                        )
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
                        bucket_counts[period][bucket] += 1
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to process CSV row {row_number}: {exc}"
                        ) from exc

                    if row_number % 10000 == 0:
                        connection.commit()
                    progress.set_postfix(
                        {
                            f"{period}:{label}": f"{selected}/{puzzles_per_bucket}"
                            for period, counts in bucket_counts.items()
                            for label, selected in zip(bucket_labels, counts)
                        },
                        refresh=False,
                    )
                    progress.update(1)
                    if all(
                        selected == puzzles_per_bucket
                        for counts in bucket_counts.values()
                        for selected in counts
                    ):
                        break

            connection.commit()
            for period, counts in bucket_counts.items():
                for label, selected in zip(bucket_labels, counts):
                    if selected < puzzles_per_bucket:
                        tqdm.write(
                            f"Input exhausted: {period} bucket {label} has "
                            f"{selected:,}/{puzzles_per_bucket:,} puzzles."
                        )
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
    parser.add_argument(
        "--puzzles-per-bucket",
        type=int,
        default=1000,
        help="Number to keep per rating bucket in each date split (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible selection",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=300,
        help="Games per API request, 1-300 (default: 300)",
    )
    parser.add_argument(
        "--date-cache",
        type=Path,
        default=None,
        help="Persistent game-date SQLite cache (default: lichess_game_dates.sqlite3 beside input)",
    )
    args = parser.parse_args()
    if args.puzzles_per_bucket <= 0:
        parser.error("--puzzles-per-bucket must be a positive integer")
    if not 1 <= args.batch_size <= 300:
        parser.error("--batch-size must be between 1 and 300")
    return args


def main() -> None:
    args = parse_args()
    token = load_lichess_token(args.secrets)
    before_count, after_count = process(
        args.input,
        args.before_output,
        args.after_output,
        token,
        args.puzzles_per_bucket,
        args.seed,
        args.date_cache,
        args.batch_size,
    )
    print(f"Wrote {before_count:,} puzzles before {CUTOFF} to {args.before_output}")
    print(f"Wrote {after_count:,} puzzles on or after {CUTOFF} to {args.after_output}")


if __name__ == "__main__":
    main()
