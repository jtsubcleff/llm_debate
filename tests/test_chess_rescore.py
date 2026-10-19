import json
import tempfile
import unittest
from pathlib import Path

import chess
import pandas as pd

from core.chess.rescore import rescore_experiment


class RescoreTests(unittest.TestCase):
    def test_preview_write_truncation_missing_traces_and_idempotence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "direct_qa" / "example" / "puzzles.csv"
            path.parent.mkdir(parents=True)
            rows = []
            for i, (text, reason, complete) in enumerate(
                [
                    ("**e4**", "STOP_SEQUENCE", True),
                    ("**e4**", "MAX_TOKENS", True),
                    ("**e4**", "REFUSAL", True),
                    ("**e4**", "STOP_SEQUENCE", False),
                    ("", "STOP_SEQUENCE", True),
                ]
            ):
                rows.append(
                    dict(
                        puzzleId=f"00{i}",
                        model="example",
                        FEN=chess.STARTING_FEN,
                        rating_bucket="Low",
                        expected_move="e4",
                        complete=complete,
                        final_move=text,
                        correct=False,
                        model_response=text,
                        tool_trace=json.dumps(
                            [dict(completion=text, stop_reason=reason)]
                        ),
                    )
                )
            rows[-1]["tool_trace"] = "[]"
            pd.DataFrame(rows).to_csv(path, index=False)
            summary = root / "results.csv"
            summary.write_text("old summary\n")
            before = path.read_bytes()
            preview = rescore_experiment(root)
            self.assertEqual(preview[0]["gained"], 1)
            self.assertEqual(preview[0]["skipped"], 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(path.parent.glob("*.bak")))
            self.assertEqual(rescore_experiment(root, write=True), preview)
            frame = pd.read_csv(path, keep_default_na=False)
            self.assertEqual(frame.correct.tolist(), [True, False, False, False, False])
            self.assertEqual(frame.final_move[:3].tolist(), ["e4", "e4", "e4"])
            self.assertEqual(
                frame.model_response.tolist(), [r["model_response"] for r in rows]
            )
            self.assertEqual(next(path.parent.glob("*.bak")).read_bytes(), before)
            self.assertEqual(pd.read_csv(summary).completed.iloc[0], 4)
            self.assertEqual(rescore_experiment(root)[0]["changed"], 0)

    def test_judge_rescore_normalizes_formatting_but_preserves_stop_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "judge" / "example" / "puzzles.csv"
            path.parent.mkdir(parents=True)
            cases = [
                ("No.", "STOP_SEQUENCE", False),
                ("**Yes**!", "STOP_SEQUENCE", True),
                ("No**", "MAX_TOKENS", False),
                ("No.", "REFUSAL", False),
                ("No. More explanation.", "STOP_SEQUENCE", False),
            ]
            rows = [
                dict(
                    puzzleId=str(i),
                    model="judge",
                    rating_bucket="Low",
                    complete=True,
                    proposer_correct=truth,
                    correct=False,
                    judgement=text.split()[-1],
                    model_response=text,
                    tool_trace=json.dumps([dict(completion=text, stop_reason=reason)]),
                )
                for i, (text, reason, truth) in enumerate(cases)
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            original = path.read_bytes()
            preview = rescore_experiment(root, protocol="judge")
            self.assertEqual(preview[0]["gained"], 2)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(
                rescore_experiment(root, write=True, protocol="judge"), preview
            )
            saved = pd.read_csv(path, keep_default_na=False)
            self.assertEqual(saved.correct.tolist(), [True, True, False, False, False])
            self.assertEqual(saved.judgement[:4].tolist(), ["No", "Yes", "No", "No"])
            self.assertEqual(
                saved.model_response.tolist(), [r["model_response"] for r in rows]
            )
            self.assertEqual(next(path.parent.glob("*.bak")).read_bytes(), original)
            self.assertEqual(
                rescore_experiment(root, protocol="judge")[0]["changed"], 0
            )
            self.assertEqual(pd.read_csv(root / "results.csv").accuracy.iloc[0], 40)


if __name__ == "__main__":
    unittest.main()
