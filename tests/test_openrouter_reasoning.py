"""Check effort configuration and the actual serialized OpenRouter payload."""

import json
import time
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from core.llm_api.openai_llm import OpenRouterChatModel
from core.llm_api.registry import catalog, resolve_model


class ReasoningConfigTests(unittest.TestCase):
    def test_optional_field_and_validation(self):
        config = catalog()
        entry = config["models"]["openrouter_gpt_oss_20b"]
        with patch("core.llm_api.registry.catalog", return_value=config):
            self.assertNotIn(
                "reasoning_effort", resolve_model("openrouter_gpt_oss_20b")
            )
            for value in (None, "low", "medium", "high"):
                entry["reasoning_effort"] = value
                self.assertEqual(
                    resolve_model("openai/gpt-oss-20b")["reasoning_effort"], value
                )
            entry["reasoning_effort"] = "typo"
            with self.assertRaisesRegex(ValueError, "Invalid reasoning_effort"):
                resolve_model("openrouter_gpt_oss_20b")
            config["models"]["anthropic_small"]["reasoning_effort"] = "low"
            with self.assertRaisesRegex(ValueError, "OpenRouter"):
                resolve_model("anthropic_small")


class ReasoningPayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_effort_defaults_routing_usage_and_explicit_overrides(self):
        for configured, override, expected in [
            (None, None, None),
            ("low", None, "low"),
            ("low", "high", "high"),
        ]:
            with self.subTest(configured=configured, override=override):
                spec = resolve_model("openrouter_gpt_oss_20b")
                spec["reasoning_effort"] = configured
                default_body = dict(provider=dict(order=["example-provider"]))
                if configured:
                    default_body["reasoning"] = dict(exclude=True)
                spec["request_defaults"] = dict(extra_body=default_body)
                requests = []

                def respond(request):
                    requests.append(json.loads(request.content))
                    return httpx.Response(
                        200,
                        json=dict(
                            id="test",
                            object="chat.completion",
                            created=0,
                            model=spec["id"],
                            usage=dict(
                                prompt_tokens=10, completion_tokens=1, total_tokens=11
                            ),
                            choices=[
                                dict(
                                    index=0,
                                    finish_reason="stop",
                                    message=dict(role="assistant", content="e4"),
                                )
                            ],
                        ),
                    )

                adapter = OpenRouterChatModel(0.9, None, api_key="test")
                await adapter.client.close()
                async with AsyncOpenAI(
                    api_key="test",
                    max_retries=0,
                    http_client=httpx.AsyncClient(
                        transport=httpx.MockTransport(respond)
                    ),
                ) as client:
                    adapter.client = client
                    extra_body = dict(custom_flag=True)
                    if override:
                        extra_body["reasoning"] = dict(effort=override)
                    with patch(
                        "core.llm_api.openai_llm.resolve_model", return_value=spec
                    ):
                        await adapter._make_api_call(
                            [dict(role="user", content="Move?")],
                            spec["id"],
                            time.time(),
                            max_tokens=100,
                            extra_body=extra_body,
                        )
                body = requests[0]
                self.assertEqual(body["usage"], dict(include=True))
                self.assertEqual(body["provider"], default_body["provider"])
                self.assertTrue(body["custom_flag"])
                self.assertNotIn("reasoning_effort", body)
                if expected:
                    self.assertEqual(
                        body["reasoning"], dict(effort=expected, exclude=True)
                    )
                else:
                    self.assertNotIn("reasoning", body)
                # Request construction must not mutate pinned config dictionaries.
                self.assertNotIn("usage", default_body)
                if configured:
                    self.assertEqual(default_body["reasoning"], dict(exclude=True))


if __name__ == "__main__":
    unittest.main()
