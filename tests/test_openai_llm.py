import json
import time
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from core.llm_api.base_llm import StopReason
from core.llm_api.openai_llm import OpenAIBaseModel, OpenAIChatModel


class OpenAISDKTests(unittest.IsolatedAsyncioTestCase):
    async def call_model(self, model_type, body, prompt, model_id, **params):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json=body)

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            model = model_type(frac_rate_limit=0.9, organization="test-org")
        await model.client.close()
        async with AsyncOpenAI(
            api_key="test-key",
            organization="test-org",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            model.client = client
            with patch("core.llm_api.openai_llm.count_tokens", return_value=2):
                responses = await model._make_api_call(
                    prompt, model_id, time.time(), **params
                )
        self.assertEqual(requests[0].headers["authorization"], "Bearer test-key")
        self.assertEqual(requests[0].headers["openai-organization"], "test-org")
        return responses, requests[0]

    async def test_chat_typed_logprobs_and_multiple_choices(self):
        # Use a mocked Chat Completions response to check logprob request parameters,
        # multiple-choice parsing, finish-reason mapping, and aggregate usage cost.
        body = {
            "id": "chat-test",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "yes"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "yes",
                                "logprob": -0.1,
                                "top_logprobs": [{"token": "yes", "logprob": -0.1}],
                            }
                        ]
                    },
                },
                {
                    "index": 1,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "no"},
                    "logprobs": None,
                },
            ],
        }
        responses, request = await self.call_model(
            OpenAIChatModel,
            body,
            [{"role": "user", "content": "test"}],
            "gpt-4",
            logprobs=2,
            n=2,
        )
        self.assertEqual(request.url.path, "/v1/chat/completions")
        self.assertEqual(json.loads(request.content)["top_logprobs"], 2)
        self.assertIs(json.loads(request.content)["logprobs"], True)
        self.assertEqual([r.completion for r in responses], ["yes", "no"])
        self.assertEqual(responses[0].logprobs, [{"yes": -0.1}])
        self.assertIsNone(responses[1].logprobs)
        self.assertEqual(responses[1].stop_reason, StopReason.MAX_TOKENS)
        # API usage covers all choices: divide the billed total, including input.
        self.assertAlmostEqual(
            sum(r.cost for r in responses), 10 * 0.03 / 1000 + 4 * 0.06 / 1000
        )

    async def test_base_completion(self):
        # Exercise the legacy Completions endpoint with a mocked response so its
        # prompt format, text, logprobs, and finish reason remain compatible.
        body = {
            "id": "completion-test",
            "object": "text_completion",
            "created": 0,
            "model": "davinci-002",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            "choices": [
                {
                    "index": 0,
                    "text": "yes",
                    "finish_reason": "stop",
                    "logprobs": {"top_logprobs": [{"yes": -0.1}]},
                }
            ],
        }
        responses, request = await self.call_model(
            OpenAIBaseModel,
            body,
            ["test"],
            "davinci-002",
            logprobs=2,
        )
        self.assertEqual(request.url.path, "/v1/completions")
        self.assertEqual(json.loads(request.content)["prompt"], ["test"])
        self.assertEqual(responses[0].completion, "yes")
        self.assertEqual(responses[0].logprobs, [{"yes": -0.1}])
        self.assertEqual(responses[0].stop_reason, StopReason.STOP_SEQUENCE)

    async def test_rate_limit_headers(self):
        # Verify that both OpenAI endpoint variants use the expected probe URL
        # and preserve rate-limit headers returned by the mocked SDK transport.
        for model_type, model_id, path in [
            (OpenAIChatModel, "gpt-4", "/v1/chat/completions"),
            (OpenAIBaseModel, "davinci-002", "/v1/completions"),
        ]:

            def respond(request):
                self.assertEqual(request.url.path, path)
                return httpx.Response(
                    200, json={}, headers={"x-ratelimit-limit-tokens": "1000"}
                )

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
                model = model_type(frac_rate_limit=0.9, organization="test-org")
            await model.client.close()
            async with AsyncOpenAI(
                api_key="test-key",
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
            ) as client:
                model.client = client
                headers = await model._get_dummy_response_header(model_id)
            self.assertEqual(headers["x-ratelimit-limit-tokens"], "1000")


if __name__ == "__main__":
    unittest.main()
