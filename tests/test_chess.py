import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import chess
import pandas as pd
from hydra import compose, initialize_config_dir

from core.chess.protocols.direct_qa import run_direct_qa, score_response
from core.chess.run import run_experiment
from core.chess.tools import validate_move_sequence
from core.llm_api.base_llm import LLMResponse
from core.llm_api.tool_calls import (
    anthropic_messages,
    assistant_message,
    responses_input,
)
from core.load.chess import ChessPuzzle, load_puzzles, rating_bucket, sample_puzzles


def response(text="e4", calls=None, reason="stop"):
    return LLMResponse(
        "gpt-4",
        text,
        reason,
        0.01,
        input_tokens=10,
        output_tokens=5,
        tool_calls=calls or [],
    )


def puzzle(identifier="p", rating=1000, fen=chess.STARTING_FEN, moves="e2e4 e7e5"):
    return ChessPuzzle(identifier, fen, moves, rating, rating_bucket(rating))


class ChessTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        self.assertEqual(
            [rating_bucket(r) for r in [1299, 1300, 2299, 2300, 2301]],
            ["Low", "Medium", "Medium", "Medium", "High"],
        )

    def test_validator_scoped_sequences_and_null_moves(self):
        original = chess.STARTING_FEN
        result = validate_move_sequence(original, ["e4", "e7e5", "Nf3"])
        board = chess.Board(original)
        for move in ["e4", "e5", "Nf3"]:
            board.push_san(move)
        self.assertEqual(
            result, dict(legal=True, fen=board.fen(), first_illegal_move=None)
        )
        self.assertTrue(validate_move_sequence(original, ["e4"])["legal"])
        for moves, illegal in [
            (["e4", "e4"], "e4"),
            (["--"], "--"),
            (["0000"], "0000"),
            (["bad"], "bad"),
        ]:
            self.assertEqual(
                validate_move_sequence(original, moves),
                dict(legal=False, fen=None, first_illegal_move=illegal),
            )
        with self.assertRaises(ValueError):
            validate_move_sequence(original, [])

    def test_final_token_exact_san(self):
        p = puzzle()
        for text, correct in [
            ("Reasoning\ne4", True),
            ("e2e4", False),
            ("e4.", False),
            ("`e4`", False),
            ("e4 extra", False),
            ("", False),
        ]:
            self.assertEqual(score_response(p, text)[1], correct)
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
        mate = puzzle(fen=board.fen(), moves="g6g7")
        self.assertEqual(mate.expected_move, "Qg7#")
        self.assertTrue(score_response(mate, "Qg7#")[1])
        promotion = puzzle(fen="7k/P7/8/8/8/8/8/7K w - - 0 1", moves="a7a8q")
        self.assertEqual(promotion.expected_move, "a8=Q+")
        castle = puzzle(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", moves="e1g1")
        self.assertEqual(castle.expected_move, "O-O")

    def test_sampling_balanced_seeded_and_shortfalls(self):
        buckets = {
            bucket: [puzzle(f"{bucket}{i}", rating) for i in range(10)]
            for bucket, rating in [("Low", 1000), ("Medium", 2000), ("High", 2500)]
        }
        selected = sample_puzzles(buckets, 9, 42)
        self.assertEqual(selected, sample_puzzles(buckets, 9, 42))
        self.assertEqual(
            pd.Series([p.rating_bucket for p in selected]).value_counts().to_dict(),
            {"Low": 3, "Medium": 3, "High": 3},
        )
        self.assertEqual(len({p.puzzleId for p in selected}), 9)
        for limit in [0, 4, -3, 33]:
            with self.assertRaises(ValueError):
                sample_puzzles(buckets, limit)

    def test_real_prepared_dataset(self):
        buckets = load_puzzles()
        self.assertTrue(all(buckets.values()))
        for p in sample_puzzles(buckets, 30):
            self.assertTrue(p.expected_move)

    def test_provider_history_preserves_reasoning(self):
        calls = [
            dict(
                id="c1",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["e4"]}'
                ),
            )
        ]
        r = response("", calls)
        native = [
            dict(type="reasoning", id="r1", summary=[]),
            dict(
                type="function_call",
                call_id="c1",
                name="validate_move_sequence",
                arguments='{"moves":["e4"]}',
            ),
        ]
        from attrs import evolve

        r = evolve(r, response_items=native, stop_reason="stop")
        messages = [
            assistant_message(r),
            dict(role="tool", tool_call_id="c1", content='{"legal":true}'),
        ]
        converted = responses_input(messages)
        self.assertEqual(converted[:2], native)
        self.assertEqual(converted[2]["type"], "function_call_output")
        a = anthropic_messages(
            [
                dict(
                    role="assistant",
                    content="",
                    response_items=[
                        dict(
                            type="tool_use",
                            id="c1",
                            name="validate_move_sequence",
                            input={},
                        )
                    ],
                ),
                messages[-1],
                dict(role="tool", tool_call_id="c2", content="result"),
            ]
        )
        self.assertEqual(len(a), 2)
        self.assertEqual(len(a[-1]["content"]), 2)


class ChessAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_tools_prompt_single_call_and_cache(self):
        api = AsyncMock(return_value=[response()])
        with tempfile.TemporaryDirectory() as folder:
            result = await run_direct_qa(
                api, "gpt-4", puzzle(), folder, use_tools=False
            )
            self.assertTrue(result["complete"])
            self.assertTrue(result["correct"])
            self.assertFalse(result["use_tools"])
            self.assertEqual(result["api_calls"], 1)
            self.assertEqual(result["input_tokens"], 10)
            self.assertEqual(result["output_tokens"], 5)
            self.assertNotIn("tools", api.call_args.kwargs)
            self.assertNotIn("tool_choice", api.call_args.kwargs)
            self.assertEqual(api.await_count, 1)
            messages = api.call_args.args[1]
            self.assertEqual(len(messages), 2)
            prompt = "\n".join(message["content"] for message in messages)
            self.assertNotIn("tool", prompt.lower())
            self.assertNotIn("validate_move_sequence", prompt)
            again = await run_direct_qa(
                AsyncMock(side_effect=AssertionError("Must reuse cache")),
                "gpt-4",
                puzzle(),
                folder,
                use_tools=False,
            )
            self.assertEqual(result, again)

    async def test_no_tools_rejects_unexpected_calls_without_execution(self):
        calls = [
            dict(
                id="c1",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["e4"]}'
                ),
            )
        ]
        api = AsyncMock(return_value=[response("", calls, "tool_calls")])
        with tempfile.TemporaryDirectory() as folder, patch(
            "core.chess.protocols.direct_qa.validate_move_sequence"
        ) as validator:
            result = await run_direct_qa(
                api, "gpt-4", puzzle(), folder, use_tools=False
            )
        validator.assert_not_called()
        self.assertEqual(api.await_count, 1)
        self.assertFalse(result["complete"])
        self.assertFalse(result["correct"])
        self.assertIn("use_tools=false", result["error"])

    async def test_tool_round_limit_usage_and_cache(self):
        calls = [
            dict(
                id="c1",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["e4"]}'
                ),
            )
        ]
        api = AsyncMock(
            side_effect=[[response("Try e4", calls, "tool_calls")], [response()]]
        )
        with tempfile.TemporaryDirectory() as folder:
            result = await run_direct_qa(
                api, "gpt-4", puzzle(), folder, max_tool_rounds=1
            )
            self.assertTrue(result["correct"])
            self.assertEqual(result["input_tokens"], 20)
            self.assertEqual(result["output_tokens"], 10)
            self.assertEqual(result["api_calls"], 2)
            self.assertEqual(api.call_args.kwargs["tool_choice"], "none")
            again = await run_direct_qa(
                AsyncMock(side_effect=AssertionError("Must use cache")),
                "gpt-4",
                puzzle(),
                folder,
                max_tool_rounds=1,
            )
            self.assertEqual(result, again)

    async def test_failure_resumes_without_rebilling_first_turn(self):
        calls = [
            dict(
                id="c1",
                type="function",
                function=dict(
                    name="validate_move_sequence", arguments='{"moves":["e4","e4"]}'
                ),
            )
        ]
        with tempfile.TemporaryDirectory() as folder:
            api = AsyncMock(
                side_effect=[
                    [response("", calls, "tool_calls")],
                    RuntimeError("transient"),
                ]
            )
            result = await run_direct_qa(api, "gpt-4", puzzle(), folder)
            self.assertFalse(result["complete"])
            retry = AsyncMock(return_value=[response()])
            result = await run_direct_qa(retry, "gpt-4", puzzle(), folder)
            self.assertTrue(result["correct"])
            self.assertEqual(retry.await_count, 1)
            self.assertEqual(result["input_tokens"], 20)
            sent = retry.call_args.args[1]
            self.assertIn(
                '"first_illegal_move": "e4"',
                next(m["content"] for m in sent if m["role"] == "tool"),
            )

    async def test_truncated_final_answer_is_wrong(self):
        with tempfile.TemporaryDirectory() as folder:
            result = await run_direct_qa(
                AsyncMock(return_value=[response(reason="length")]),
                "gpt-4",
                puzzle(),
                folder,
            )
            self.assertTrue(result["complete"])
            self.assertFalse(result["correct"])

    async def test_multi_model_checkpoint_resume_and_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "puzzles.csv"
            with source.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["PuzzleId", "FEN", "Moves", "Rating"])
                for i, rating in enumerate([1000, 2000, 2500]):
                    writer.writerow([f"000{i}", chess.STARTING_FEN, "e2e4", rating])
            with initialize_config_dir(
                version_base=None, config_dir=str(Path("core/config").resolve())
            ):
                cfg = compose(
                    config_name="chess",
                    overrides=[
                        f"exp_dir={folder}/exp",
                        f"dataset_path={source}",
                        "models=[gpt-4,gpt-3.5-turbo]",
                        "limit=3",
                    ],
                )
            api = AsyncMock(return_value=[response()])
            frames = await run_experiment(cfg, api)
            self.assertEqual(api.await_count, 6)
            self.assertEqual(len(frames), 2)
            summary = pd.read_csv(Path(cfg.exp_dir) / "results.csv")
            self.assertTrue(
                (summary[["Low", "Medium", "High", "all"]] == 1).all().all()
            )
            self.assertEqual(summary.completed.tolist(), [3, 3])
            await run_experiment(
                cfg,
                AsyncMock(side_effect=AssertionError("Must skip completed puzzles")),
            )
            # Older manifests implicitly used tools and remain resumable.
            manifest = Path(cfg.exp_dir) / "chess_run.json"
            old_settings = json.loads(manifest.read_text())
            old_settings.pop("use_tools")
            manifest.write_text(json.dumps(old_settings))
            await run_experiment(
                cfg,
                AsyncMock(side_effect=AssertionError("Must skip completed puzzles")),
            )
            # Changing mode cannot reuse tool-enabled results or caches.
            cfg.use_tools = False
            with self.assertRaisesRegex(ValueError, "settings"):
                await run_experiment(cfg, api)
            cfg.exp_dir = str(Path(folder) / "no_tools")
            no_tools_api = AsyncMock(return_value=[response()])
            no_tools_frames = await run_experiment(cfg, no_tools_api)
            self.assertEqual(no_tools_api.await_count, 6)
            self.assertTrue(
                all(not frame.use_tools.any() for frame in no_tools_frames.values())
            )
            for call in no_tools_api.call_args_list:
                self.assertNotIn("tools", call.kwargs)
                self.assertNotIn("tool_choice", call.kwargs)
            settings = json.loads((Path(cfg.exp_dir) / "chess_run.json").read_text())
            self.assertFalse(settings["use_tools"])
            self.assertIsNone(settings["tool"])
            self.assertNotIn("validate_move_sequence", json.dumps(settings["prompt"]))
            cfg.temperature = 0.8
            with self.assertRaisesRegex(ValueError, "settings"):
                await run_experiment(cfg, api)


if __name__ == "__main__":
    unittest.main()
