import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from openai import RateLimitError

from core.llm_api.anthropic_llm import AnthropicChatModel
from core.llm_api.base_llm import LLMResponse
from core.llm_api.errors import is_credit_exhaustion
from core.llm_api.openai_llm import OpenAIChatModel
from core.llm_api.registry import model_id


def rate_limit_error(code, error_type, message="Request denied"):
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    return RateLimitError(
        message,
        response=response,
        body={
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        },
    )


class CreditRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_credit_and_quota_errors_do_not_retry_openai(self):
        # Simulate both OpenAI billing-exhaustion codes and require the original
        # exception after one attempt, with no retry delay despite a budget of ten attempts.
        for code, kind in [
            ("credit_balance_exhausted", "insufficient_quota"),
            ("insufficient_quota", "insufficient_quota"),
        ]:
            with self.subTest(code=code), patch.dict(
                os.environ, {"OPENAI_API_KEY": "test-key"}
            ):
                adapter = OpenAIChatModel(frac_rate_limit=0.9, organization=None)
                error = rate_limit_error(code, kind, "You have no credits remaining.")
                call = AsyncMock(side_effect=error)
                with patch.object(
                    OpenAIChatModel, "_make_api_call", call
                ), patch.object(
                    OpenAIChatModel,
                    "_create_prompt_history_file",
                    return_value="unused",
                ), patch(
                    "core.llm_api.openai_llm.asyncio.sleep", new_callable=AsyncMock
                ) as sleep:
                    with self.assertRaises(RateLimitError) as caught:
                        await adapter(
                            [model_id("judge_4o_mini")],
                            [{"role": "user", "content": "test"}],
                            False,
                            10,
                        )
                    self.assertIs(caught.exception, error)
                    self.assertEqual(call.await_count, 1)
                    sleep.assert_not_awaited()
                await adapter.client.close()

    async def test_temporary_rate_limit_still_retries(self):
        # Simulate a temporary rate limit followed by success. This guards against
        # accidentally treating every HTTP 429 response as a permanent billing failure.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIChatModel(frac_rate_limit=0.9, organization=None)
        result = LLMResponse(
            model_id=model_id("judge_4o_mini"),
            completion="A",
            stop_reason="stop",
            cost=0,
        )
        call = AsyncMock(
            side_effect=[rate_limit_error("rate_limit_exceeded", "requests"), [result]]
        )
        with patch.object(OpenAIChatModel, "_make_api_call", call), patch.object(
            OpenAIChatModel, "_create_prompt_history_file", return_value="unused"
        ), patch.object(OpenAIChatModel, "_add_response_to_prompt_file"), patch(
            "core.llm_api.openai_llm.LOGGER"
        ), patch(
            "core.llm_api.openai_llm.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            responses = await adapter(
                [model_id("judge_4o_mini")],
                [{"role": "user", "content": "test"}],
                False,
                10,
            )
            self.assertEqual(responses, [result])
            self.assertEqual(call.await_count, 2)
            sleep.assert_awaited_once()
        await adapter.client.close()

    async def test_header_probe_does_not_retry_billing_errors(self):
        # Check that the separately decorated rate-limit header probe also stops
        # after one billing failure instead of retrying through Tenacity.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            adapter = OpenAIChatModel(frac_rate_limit=0.9, organization=None)
        error = rate_limit_error("credit_balance_exhausted", "insufficient_quota")
        with patch.object(
            adapter.client.chat.completions.with_raw_response,
            "create",
            new=AsyncMock(side_effect=error),
        ) as call:
            with self.assertRaises(RateLimitError):
                await adapter._get_dummy_response_header(model_id("judge_4o_mini"))
            self.assertEqual(call.await_count, 1)
        await adapter.client.close()

    async def test_anthropic_credit_message_does_not_retry(self):
        # Simulate a credit-denial message without structured error codes to verify
        # the Anthropic adapter propagates it immediately without sleeping or retrying.
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            adapter = AnthropicChatModel(num_threads=1)
        error = RuntimeError("Your credit balance is too low to access the API.")
        with patch.object(
            adapter.client.messages, "create", new=AsyncMock(side_effect=error)
        ) as call, patch(
            "core.llm_api.anthropic_llm.asyncio.sleep", new_callable=AsyncMock
        ) as sleep:
            with self.assertRaises(RuntimeError) as caught:
                await adapter(
                    [model_id("anthropic_balanced")],
                    [{"role": "user", "content": "test"}],
                    False,
                    10,
                )
            self.assertIs(caught.exception, error)
            self.assertEqual(call.await_count, 1)
            sleep.assert_not_awaited()
        await adapter.client.close()

    def test_nested_body_and_normal_rate_limit_classification(self):
        # Recognize billing exhaustion inside a nested error payload while keeping
        # ordinary token rate limits classified as retryable.
        error = RuntimeError("Request denied")
        error.body = {"error": {"code": "credit_balance_exhausted"}}
        self.assertTrue(is_credit_exhaustion(error))
        self.assertFalse(
            is_credit_exhaustion(rate_limit_error("rate_limit_exceeded", "tokens"))
        )


if __name__ == "__main__":
    unittest.main()
