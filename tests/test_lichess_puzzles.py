"""Offline checks for puzzle selection, compressed input, and batched date lookups."""

import csv
import io
import json
import random
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

import chess
import zstandard

from scripts import filter_lichess_puzzles as puzzles
from scripts import prepare_lichess_puzzles as prepare


class QuietProgress:
    def __init__(self, iterable=(), **kwargs):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        return iter(self.iterable)

    def update(self, *args):
        pass

    def set_description(self, *args):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    @staticmethod
    def write(message):
        pass


class PuzzleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "puzzles.csv.zst"
        self.rows = []
        for repeat in range(3):
            for period in ("A", "B"):
                for bucket, rating in enumerate((1299, 1300, 2300)):
                    game_id = f"{period}{repeat}{bucket}"
                    self.rows.append(
                        [
                            game_id,
                            chess.STARTING_FEN,
                            "e2e4 e7e5 g1f3",
                            rating,
                            "",
                            "",
                            "",
                            "theme",
                            f"https://lichess.org/{game_id}#2",
                            "",
                            "",
                        ]
                    )
        self.write_source(self.rows)
        for module in (puzzles, prepare):
            quiet = patch.object(module, "tqdm", QuietProgress)
            quiet.start()
            self.addCleanup(quiet.stop)

    def write_source(self, rows):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(puzzles.INPUT_COLUMNS)
        for row in rows:
            writer.writerow(row)
            writer.writerow([])  # Blank records must not change random offsets.
        self.source.write_bytes(
            zstandard.ZstdCompressor().compress(stream.getvalue().encode())
        )

    @staticmethod
    def dates(game_ids, token):
        return {
            game_id: date(2024, 5, 31) if game_id.startswith("A") else date(2024, 6, 1)
            for game_id in reversed(game_ids)
        }

    def read_output(self, path):
        with path.open(newline="") as source:
            reader = csv.DictReader(source)
            self.assertEqual(reader.fieldnames, puzzles.OUTPUT_COLUMNS)
            return list(reader)

    def test_batch_export_matches_ids_and_preserves_original_dates(self):
        pgn = (
            '[Site "https://lichess.org/newGame1"]\n[Date "2024.06.01"]\n\n*\n\n'
            '[Site "https://lichess.org/oldGame1"]\n[Date "2024.05.31"]\n\n*\n'
        )
        with patch.object(
            puzzles, "urlopen", return_value=io.BytesIO(pgn.encode())
        ) as request:
            result = puzzles.fetch_game_dates(["oldGame1", "newGame1"], "fake-token")
        self.assertEqual(
            result, {"oldGame1": date(2024, 5, 31), "newGame1": date(2024, 6, 1)}
        )
        sent = request.call_args.args[0]
        self.assertEqual(sent.get_method(), "POST")
        self.assertIn("/api/games/export/_ids?moves=false", sent.full_url)
        self.assertEqual(sent.data, b"oldGame1,newGame1")
        self.assertEqual(sent.get_header("Authorization"), "Bearer fake-token")

    def test_missing_games_fail_instead_of_silently_dropping_rows(self):
        with patch.object(puzzles, "download_pgn", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "missing1"):
                puzzles.fetch_game_dates(["missing1"], "fake-token")

    def test_rate_limit_waits_one_minute(self):
        error = HTTPError("https://lichess.org/", 429, "Too many requests", None, None)
        with patch.object(
            puzzles, "urlopen", side_effect=[error, io.BytesIO(b"PGN")]
        ), patch.object(puzzles.time, "sleep") as sleep:
            self.assertEqual(
                puzzles.download_pgn(Request("https://lichess.org/")), "PGN"
            )
        sleep.assert_called_once_with(60)

    def test_random_rotation_and_count_cache_invalidation(self):
        def read(seed):
            with puzzles.randomized_source_rows(self.source, seed) as rows:
                return [(number, row["PuzzleId"]) for number, row in rows]

        first = read(42)
        self.assertEqual(first, read(42))
        self.assertNotEqual(first, read(43))
        start = random.Random(42).randrange(len(self.rows))
        expected = self.rows[start:] + self.rows[:start]
        self.assertEqual(
            [identifier for _, identifier in first], [row[0] for row in expected]
        )
        self.assertEqual(len({identifier for _, identifier in first}), len(self.rows))
        with patch.object(
            puzzles, "tqdm", side_effect=AssertionError("Recounted unchanged input")
        ):
            self.assertEqual(puzzles.source_row_count(self.source), len(self.rows))
        self.write_source(self.rows[:4])
        self.assertEqual(puzzles.source_row_count(self.source), 4)
        sidecar = self.source.with_name(self.source.name + ".rows.json")
        sidecar.write_text("invalid json")
        self.assertEqual(puzzles.source_row_count(self.source), 4)
        self.assertEqual(json.loads(sidecar.read_text())["rows"], 4)

    def test_seeded_quotas_batching_and_persistent_cache(self):
        before, after = self.root / "before.csv", self.root / "after.csv"
        with patch.object(puzzles, "fetch_game_dates", side_effect=self.dates) as fetch:
            result = puzzles.process(self.source, before, after, "fake-token", 2, 42)
        self.assertEqual(result, (6, 6))
        self.assertLess(fetch.call_count, 12)
        first = before.read_bytes(), after.read_bytes()
        for path, prefix in ((before, "A"), (after, "B")):
            rows = self.read_output(path)
            self.assertEqual(
                [int(row["Rating"]) for row in rows],
                [1299, 1299, 1300, 1300, 2300, 2300],
            )
            self.assertTrue(all(row["PuzzleId"].startswith(prefix) for row in rows))
            board = chess.Board()
            board.push_uci("e2e4")
            for row in rows:
                self.assertEqual(row["FEN"], board.fen())
                self.assertEqual(row["Moves"], "e7e5 g1f3")
        with patch.object(
            puzzles, "fetch_game_dates", side_effect=AssertionError("Repeated API call")
        ):
            self.assertEqual(
                puzzles.process(self.source, before, after, "fake-token", 2, 42), result
            )
        self.assertEqual((before.read_bytes(), after.read_bytes()), first)
        # A different batch size must select the same puzzles, including with a warm cache.
        with patch.object(
            puzzles, "fetch_game_dates", side_effect=AssertionError("Repeated API call")
        ):
            puzzles.process(
                self.source, before, after, "fake-token", 2, 42, batch_size=1
            )
        self.assertEqual((before.read_bytes(), after.read_bytes()), first)

    def test_six_hundred_games_use_two_api_requests(self):
        rows = []
        for index in range(600):
            row = self.rows[index % 6].copy()
            row[0] = f"{row[0]}_{index}"
            row[8] = f"https://lichess.org/{row[0]}"
            rows.append(row)
        self.write_source(rows)
        with patch.object(puzzles, "fetch_game_dates", side_effect=self.dates) as fetch:
            result = puzzles.process(
                self.source,
                self.root / "a.csv",
                self.root / "b.csv",
                "fake-token",
                100,
                42,
            )
        self.assertEqual(result, (300, 300))
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(
            [len(call.args[0]) for call in fetch.call_args_list], [300, 300]
        )

    def test_successful_dates_survive_later_move_validation_failure(self):
        bad = [row.copy() for row in self.rows]
        for row in bad:
            row[2] = "e2e5"  # Illegal first move.
        self.write_source(bad)
        with patch.object(puzzles, "fetch_game_dates", side_effect=self.dates):
            with self.assertRaisesRegex(RuntimeError, "invalid first move"):
                puzzles.process(
                    self.source,
                    self.root / "a.csv",
                    self.root / "b.csv",
                    "fake-token",
                    1,
                    42,
                )
        with sqlite3.connect(str(self.root / "lichess_game_dates.sqlite3")) as cache:
            self.assertGreater(
                cache.execute("SELECT COUNT(*) FROM game_dates").fetchone()[0], 0
            )

    def test_shortfalls_and_shared_no_api_workflow(self):
        self.write_source([row for row in self.rows if row[0].startswith("A")])
        with patch.object(puzzles, "fetch_game_dates", side_effect=self.dates):
            result = puzzles.process(
                self.source,
                self.root / "a.csv",
                self.root / "b.csv",
                "fake-token",
                1,
                42,
            )
        self.assertEqual(result, (3, 0))
        with patch.object(
            puzzles,
            "fetch_game_dates",
            side_effect=AssertionError("Unexpected API call"),
        ):
            self.assertEqual(
                prepare.process(self.source, self.root / "all.csv", 1, 42), 3
            )


if __name__ == "__main__":
    unittest.main()
