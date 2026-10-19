import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from core.chess.results import print_run_cost, write_csv


class CostSummaryTests(unittest.TestCase):
    def test_saved_usage_includes_failures_unknowns_and_both_stages_once(self):
        with tempfile.TemporaryDirectory() as folder:
            proposer = pd.DataFrame(
                [
                    dict(
                        model="proposer",
                        complete=True,
                        cost=0.125,
                        input_tokens=100,
                        output_tokens=20,
                        api_calls=1,
                    ),
                    dict(
                        model="proposer",
                        complete=False,
                        cost=0.025,
                        input_tokens=40,
                        output_tokens=10,
                        api_calls=1,
                    ),
                    dict(
                        model="proposer",
                        complete=False,
                        cost="",
                        input_tokens="",
                        output_tokens="",
                        api_calls="",
                    ),
                ]
            )
            judge = pd.DataFrame(
                [
                    dict(
                        model="judge",
                        complete=True,
                        cost="",
                        input_tokens=90,
                        output_tokens=30,
                        api_calls=1,
                    ),
                    dict(
                        model="judge",
                        complete=True,
                        cost=0.05,
                        input_tokens=30,
                        output_tokens=10,
                        api_calls=1,
                    ),
                ]
            )
            write_csv(proposer, Path(folder) / "direct_qa/p/puzzles.csv")
            write_csv(judge, Path(folder) / "judgements/judge/j/puzzles.csv")
            outputs = []
            for _ in range(2):
                output = io.StringIO()
                with redirect_stdout(output):
                    print_run_cost(folder, "proposer_judge")
                outputs.append(output.getvalue())
            self.assertEqual(outputs[0], outputs[1])
            self.assertIn(
                "proposer: $0.150000 | 140 input / 30 output tokens", outputs[0]
            )
            self.assertIn(
                "judge: $0.050000 + unknown costs for 1 puzzle(s)", outputs[0]
            )
            self.assertIn(
                "Total estimated cost (USD): $0.200000 + unknown costs for 1 puzzle(s)",
                outputs[0],
            )

    def test_no_output_without_saved_results(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(
            io.StringIO()
        ) as output:
            print_run_cost(folder)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
