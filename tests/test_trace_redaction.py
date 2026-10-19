import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, mock_open, patch

import chess

from core.chess.protocols.direct_qa import run_direct_qa
from core.llm_api.base_llm import LLMResponse
from core.llm_api.openai_llm import OpenAIChatModel
from core.llm_api.redaction import REDACTED, redact_opaque_data
from core.load.chess import ChessPuzzle


class TraceRedactionTests(unittest.TestCase):
    def test_nested_payloads_redacted_without_mutation(self):
        payload = dict(
            request_id="req_test",
            content=[
                dict(
                    type="thinking",
                    thinking="Readable thought",
                    signature="claude_blob",
                ),
                dict(type="reasoning", encrypted_content="openai_blob", summary=[]),
                dict(type="redacted_thinking", data="redacted_blob"),
                dict(type="reasoning.encrypted", data="router_blob"),
                dict(
                    type="tool_use",
                    name="validate_move_sequence",
                    input=dict(moves=["e4"]),
                ),
            ],
        )
        before = copy.deepcopy(payload)
        logged = redact_opaque_data(payload)
        self.assertEqual(payload, before)
        for blob in ("claude_blob", "openai_blob", "redacted_blob", "router_blob"):
            self.assertNotIn(blob, json.dumps(logged))
        self.assertEqual(logged["content"][0]["thinking"], "Readable thought")
        self.assertEqual(logged["content"][-1], payload["content"][-1])
        self.assertEqual(logged["request_id"], "req_test")

    def test_prompt_and_response_history_are_sanitized(self):
        native = [dict(type="reasoning", encrypted_content="opaque_blob", summary=[])]
        response = LLMResponse("gpt-4", "e4", "stop", 0, response_items=native)
        self.assertEqual(
            response.to_dict()["response_items"][0]["encrypted_content"], REDACTED
        )
        self.assertEqual(
            response.to_dict(include_opaque=True)["response_items"], native
        )
        handle = mock_open()
        with patch("builtins.open", handle):
            filename = OpenAIChatModel._create_prompt_history_file(
                [dict(role="assistant", response_items=native)]
            )
            OpenAIChatModel._add_response_to_prompt_file(filename, [response])
        written = "".join(call.args[0] for call in handle().write.call_args_list)
        self.assertNotIn("opaque_blob", written)
        self.assertIn(REDACTED, written)
        self.assertEqual(native[0]["encrypted_content"], "opaque_blob")


class TraceReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_replay_keeps_opaque_state_but_result_trace_does_not(self):
        native = [dict(type="thinking", thinking="", signature="needed_for_replay")]
        call = dict(
            id="c1",
            type="function",
            function=dict(name="validate_move_sequence", arguments='{"moves":["e4"]}'),
        )
        first = LLMResponse(
            "gpt-4", "", "tool_calls", 0, tool_calls=[call], response_items=native
        )
        final = LLMResponse("gpt-4", "e4", "stop", 0)
        api = AsyncMock(side_effect=[[first], [final]])
        puzzle = ChessPuzzle("test", chess.STARTING_FEN, "e2e4", 1000, "Low")
        with tempfile.TemporaryDirectory() as folder:
            result = await run_direct_qa(api, "gpt-4", puzzle, folder)
            self.assertNotIn("needed_for_replay", result["tool_trace"])
            self.assertIn(REDACTED, result["tool_trace"])
            self.assertEqual(api.call_args.args[1][2]["response_items"], native)
            cached = [p.read_text() for p in Path(folder).glob("*.json")]
            self.assertTrue(any("needed_for_replay" in text for text in cached))
            replay = await run_direct_qa(
                AsyncMock(side_effect=AssertionError("Must use cache")),
                "gpt-4",
                puzzle,
                folder,
            )
            self.assertEqual(result, replay)


if __name__ == "__main__":
    unittest.main()
