import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from core.llm_api.llm import ModelAPI
from core.llm_api.openai_llm import OpenAIChatModel, OpenRouterChatModel
from core.llm_api.registry import resolve_model


class OpenRouterTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_outer_and_sdk_timeouts_match(self):
        original_wait_for = asyncio.wait_for
        for adapter_type, model_id, configured, override, expected in [
            (OpenRouterChatModel, "openai/gpt-oss-20b", None, None, 600),
            (OpenRouterChatModel, "openai/gpt-oss-20b", 300, None, 300),
            (OpenRouterChatModel, "openai/gpt-oss-20b", 600, 450, 450),
            (OpenAIChatModel, "gpt-4", None, None, 120),
        ]:
            with self.subTest(model=model_id, configured=configured, override=override):

                def respond(request):
                    self.assertEqual(request.extensions["timeout"]["read"], expected)
                    body = json.loads(request.content)
                    self.assertNotIn("timeout", body)
                    return httpx.Response(
                        200,
                        json=dict(
                            id="test",
                            object="chat.completion",
                            created=0,
                            model=model_id,
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

                kwargs = {} if configured is None else dict(request_timeout=configured)
                adapter = adapter_type(0.9, None, api_key="test", **kwargs)
                await adapter.client.close()
                async with AsyncOpenAI(
                    api_key="test",
                    max_retries=0,
                    http_client=httpx.AsyncClient(
                        transport=httpx.MockTransport(respond)
                    ),
                ) as client:
                    adapter.client = client

                    async def wait_for(awaitable, timeout):
                        self.assertEqual(timeout, expected)
                        return await original_wait_for(awaitable, timeout)

                    with patch(
                        "core.llm_api.openai_llm.asyncio.wait_for", side_effect=wait_for
                    ), patch.object(
                        adapter_type,
                        "_create_prompt_history_file",
                        return_value="unused",
                    ), patch.object(
                        adapter_type, "_add_response_to_prompt_file"
                    ):
                        params = {} if override is None else dict(timeout=override)
                        result = await adapter(
                            [model_id],
                            [dict(role="user", content="Move?")],
                            False,
                            1,
                            max_tokens=100,
                            **params
                        )
                    self.assertEqual(result[0].completion, "e4")

    async def test_model_api_passes_configuration_to_openrouter_only(self):
        with patch(
            "core.utils.load_secrets", return_value={"OPENROUTER_API_KEY": "test"}
        ):
            api = ModelAPI(openrouter_timeout=360)
            try:
                adapter = api._adapter(resolve_model("openrouter_gpt_oss_20b"))
                self.assertEqual(adapter.request_timeout, 360)
            finally:
                await api.close()
        for value in (0, -1):
            with self.assertRaises(ValueError):
                ModelAPI(openrouter_timeout=value)


if __name__ == "__main__":
    unittest.main()
