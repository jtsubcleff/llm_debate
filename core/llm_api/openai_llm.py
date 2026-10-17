import asyncio
import json
import logging
import os
import time
from datetime import datetime
from itertools import cycle
from traceback import format_exc
from typing import Optional, Union

import attrs
import tiktoken
from openai import AsyncOpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChoiceLogprobs
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from termcolor import cprint

from core.llm_api.base_llm import (
    PRINT_COLORS,
    LLMResponse,
    ModelAPIProtocol,
    messages_to_single_prompt,
)
from core.llm_api.errors import is_credit_exhaustion
from core.llm_api.registry import estimate_cost
from core.llm_api.registry import price_per_token as registry_price
from core.llm_api.registry import resolve_model

OAIChatPrompt = list[dict[str, str]]
OAIBasePrompt = Union[str, list[str]]
LOGGER = logging.getLogger(__name__)


def count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def price_per_token(model_id: str) -> tuple[float, float]:
    # Unknown pricing must not prevent routing a newly configured model.
    return registry_price(model_id) or (float("inf"), float("inf"))


@attrs.define()
class Resource:
    """
    A resource that is consumed over time and replenished at a constant rate.
    """

    refresh_rate: float = (
        attrs.field()
    )  # How many units of the resource are replenished per minute
    value: float = attrs.field(init=False)
    total: float = 0
    throughput: float = 0
    last_update_time: float = attrs.field(init=False, factory=time.time)
    start_time: float = attrs.field(init=False, factory=time.time)

    def __attrs_post_init__(self):
        self.value = self.refresh_rate

    def _replenish(self):
        """
        Updates the value of the resource based on the time since the last update.
        """
        curr_time = time.time()
        self.value = min(
            self.refresh_rate,
            self.value + (curr_time - self.last_update_time) * self.refresh_rate / 60,
        )
        self.last_update_time = curr_time
        self.throughput = self.total / (curr_time - self.start_time) * 60

    def geq(self, amount: float) -> bool:
        self._replenish()
        return self.value >= amount

    def consume(self, amount: float):
        """
        Consumes the given amount of the resource.
        """
        assert self.geq(
            amount
        ), f"Resource does not have enough capacity to consume {amount} units"
        self.value -= amount
        self.total += amount


@attrs.define
class OpenAIModel(ModelAPIProtocol):
    frac_rate_limit: float
    organization: Optional[str]
    print_prompt_and_response: bool = False
    model_ids: set[str] = attrs.field(init=False, default=attrs.Factory(set))

    client: AsyncOpenAI = attrs.field(
        init=False,
        default=attrs.Factory(
            lambda self: AsyncOpenAI(organization=self.organization, max_retries=0),
            takes_self=True,
        ),
    )

    # rate limit
    token_capacity: dict[str, Resource] = attrs.field(
        init=False, default=attrs.Factory(dict)
    )
    request_capacity: dict[str, Resource] = attrs.field(
        init=False, default=attrs.Factory(dict)
    )
    lock_add: asyncio.Lock = attrs.field(
        init=False, default=attrs.Factory(asyncio.Lock)
    )
    lock_consume: asyncio.Lock = attrs.field(
        init=False, default=attrs.Factory(asyncio.Lock)
    )

    @staticmethod
    def _assert_valid_id(model_id: str):
        raise NotImplementedError

    @staticmethod
    async def _get_dummy_response_header(model_id: str):
        raise NotImplementedError

    @staticmethod
    def _count_prompt_token_capacity(prompt, **kwargs) -> int:
        raise NotImplementedError

    async def _make_api_call(self, prompt, model_id, **params) -> list[LLMResponse]:
        raise NotImplementedError

    @staticmethod
    def _print_prompt_and_response(prompt, responses):
        raise NotImplementedError

    @staticmethod
    def _create_prompt_history_file(prompt):
        filename = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}_prompt.txt"
        with open(os.path.join("prompt_history", filename), "w") as f:
            json_str = json.dumps(prompt, indent=4)
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

        return filename

    @staticmethod
    def _add_response_to_prompt_file(prompt_file, responses):
        with open(os.path.join("prompt_history", prompt_file), "a") as f:
            f.write("\n\n======RESPONSE======\n\n")
            json_str = json.dumps(
                [response.to_dict() for response in responses], indent=4
            )
            json_str = json_str.replace("\\n", "\n")
            f.write(json_str)

    async def add_model_id(self, model_id: str):
        self._assert_valid_id(model_id)
        if model_id in self.model_ids:
            return

        self.model_ids.add(model_id)

        # Configured limits avoid an extra billed probe for every new model.
        # SPECIAL TREATMENT: provider/account limits are independent of model capability.
        limits = resolve_model(model_id)
        token_capacity = Resource(
            limits.get("tokens_per_minute", 1_000_000) * self.frac_rate_limit
        )
        request_capacity = Resource(
            limits.get("requests_per_minute", 60) * self.frac_rate_limit
        )
        self.token_capacity[model_id] = token_capacity
        self.request_capacity[model_id] = request_capacity

    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        start = time.time()

        async def attempt_api_call():
            for model_id in cycle(model_ids):
                async with self.lock_consume:
                    request_capacity, token_capacity = (
                        self.request_capacity[model_id],
                        self.token_capacity[model_id],
                    )
                    if request_capacity.geq(1) and token_capacity.geq(token_count):
                        request_capacity.consume(1)
                        token_capacity.consume(token_count)
                    else:
                        await asyncio.sleep(0.01)
                        continue  # Skip this iteration if the condition isn't met

                # Make the API call outside the lock
                return await asyncio.wait_for(
                    self._make_api_call(prompt, model_id, start, **kwargs),
                    timeout=kwargs.get("timeout", 120),
                )

        model_ids.sort(
            key=lambda model_id: price_per_token(model_id)[0]
        )  # Default to cheapest model
        async with self.lock_add:
            for model_id in model_ids:
                await self.add_model_id(model_id)
        prompt = self._process_prompt(prompt)

        token_count = self._count_prompt_token_capacity(prompt, **kwargs)
        assert (
            max(self.token_capacity[model_id].refresh_rate for model_id in model_ids)
            >= token_count
        ), "Prompt is too long for any model to handle."
        prompt_file = self._create_prompt_history_file(prompt)
        responses: Optional[list[LLMResponse]] = None
        for i in range(max_attempts):
            try:
                responses = await attempt_api_call()
            except Exception as e:
                if is_credit_exhaustion(e):
                    raise
                if isinstance(e, (ValueError, TypeError)) or getattr(
                    e, "status_code", None
                ) in (400, 401, 403, 404, 422):
                    raise
                error_info = f"Exception Type: {type(e).__name__}, Error Details: {str(e)}, Traceback: {format_exc()}"
                LOGGER.warn(
                    f"Encountered API error: {error_info}.\nRetrying now. (Attempt {i})"
                )
                await asyncio.sleep(1.5**i)
            else:
                break

        if responses is None:
            raise RuntimeError(
                f"Failed to get a response from the API after {max_attempts} attempts."
            )

        self._add_response_to_prompt_file(prompt_file, responses)
        if self.print_prompt_and_response or print_prompt_and_response:
            self._print_prompt_and_response(prompt, responses)

        end = time.time()
        LOGGER.debug(f"Completed call in {end - start}s.")
        return responses


class OpenAIChatModel(OpenAIModel):
    def _process_prompt(self, prompt: OAIChatPrompt) -> OAIChatPrompt:
        return prompt

    def _assert_valid_id(self, model_id: str):
        spec = resolve_model(model_id)
        if spec["provider"] != "openai" or spec["endpoint"] not in (
            "chat",
            "responses",
        ):
            raise ValueError(f"Not an OpenAI chat/responses model: {model_id}")

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_fixed(2),
        retry=retry_if_exception(lambda error: not is_credit_exhaustion(error)),
    )
    async def _get_dummy_response_header(self, model_id: str):
        response = await self.client.chat.completions.with_raw_response.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say 1"}],
        )
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIChatPrompt, **kwargs) -> int:
        # The magic formula is: .25 * (total number of characters) + (number of messages) + (max_tokens, or 15 if not specified)
        BUFFER = 5  # A bit of buffer for some error margin
        MIN_NUM_TOKENS = 20

        num_tokens = 0
        for message in prompt:
            num_tokens += 1
            num_tokens += len(message["content"]) / 4

        return max(
            MIN_NUM_TOKENS,
            int(num_tokens + BUFFER)
            + kwargs.get("n", 1) * kwargs.get("max_tokens", 15),
        )

    def convert_top_logprobs(self, data: ChoiceLogprobs):
        return [
            {entry.token: entry.logprob for entry in item.top_logprobs}
            for item in (data.content or [])
        ]

    async def _make_api_call(
        self, prompt: OAIChatPrompt, model_id, start_time, **params
    ) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")

        spec = resolve_model(model_id)
        # SPECIAL TREATMENT: reasoning models reject sampling parameters, and
        # Responses uses different output/token fields from Chat Completions.
        params = {**spec.get("request_defaults", {}), **params}
        max_tokens = params.pop("max_tokens", None) or spec["default_max_tokens"]
        if not spec.get("sampling", False):
            params.pop("temperature", None)
            params.pop("top_p", None)
        if "logprobs" in params and not spec.get("logprobs", False):
            raise ValueError(
                f"{model_id} does not support logprobs in its configured profile"
            )
        api_start = time.time()
        if spec["endpoint"] == "responses":
            n = params.pop("n", 1)
            if n != 1:
                raise ValueError("Responses requires one request per completion")
            response = await self.client.responses.create(
                model=model_id, input=prompt, max_output_tokens=max_tokens, **params
            )
            if response.status not in ("completed", "incomplete"):
                raise RuntimeError(
                    f"Responses request ended with status {response.status}: {response.error}"
                )
            reason = "completed"
            if response.status == "incomplete":
                reason = getattr(response.incomplete_details, "reason", None)
                if reason not in ("max_output_tokens", "content_filter"):
                    raise RuntimeError(f"Incomplete response: {reason}")
                reason = "max_tokens" if reason == "max_output_tokens" else reason
            return [
                LLMResponse(
                    model_id=model_id,
                    completion=response.output_text,
                    stop_reason=reason,
                    duration=time.time() - start_time,
                    api_duration=time.time() - api_start,
                    cost=estimate_cost(
                        model_id,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    ),
                )
            ]
        if "logprobs" in params:
            params["top_logprobs"] = params["logprobs"]
            params["logprobs"] = True
        api_response = await self.client.chat.completions.create(
            messages=prompt, model=model_id, max_tokens=max_tokens, **params
        )
        # Split billed usage across choices; aggregate cost counts every token once.
        cost = estimate_cost(
            model_id,
            api_response.usage.prompt_tokens,
            api_response.usage.completion_tokens,
        )
        return [
            LLMResponse(
                model_id=model_id,
                completion=choice.message.content or "",
                stop_reason=choice.finish_reason,
                api_duration=time.time() - api_start,
                duration=time.time() - start_time,
                cost=None if cost is None else cost / len(api_response.choices),
                logprobs=self.convert_top_logprobs(choice.logprobs)
                if choice.logprobs is not None
                else None,
            )
            for choice in api_response.choices
        ]

    @staticmethod
    def _print_prompt_and_response(
        prompts: OAIChatPrompt, responses: list[LLMResponse]
    ):
        for prompt in prompts:
            role, text = prompt["role"], prompt["content"]
            cprint(f"=={role.upper()}:", "white")
            cprint(text, PRINT_COLORS[role])
        for i, response in enumerate(responses):
            if len(responses) > 1:
                cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()


class OpenAIBaseModel(OpenAIModel):
    def _process_prompt(
        self, prompt: Union[OAIBasePrompt, OAIChatPrompt]
    ) -> OAIBasePrompt:
        if isinstance(prompt, list) and isinstance(prompt[0], dict):
            return messages_to_single_prompt(prompt)
        return prompt

    def _assert_valid_id(self, model_id: str):
        spec = resolve_model(model_id)
        if spec["provider"] != "openai" or spec["endpoint"] != "completions":
            raise ValueError(f"Not an OpenAI completion model: {model_id}")

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_fixed(2),
        retry=retry_if_exception(lambda error: not is_credit_exhaustion(error)),
    )
    async def _get_dummy_response_header(self, model_id: str):
        response = await self.client.completions.with_raw_response.create(
            model=model_id, prompt="a", max_tokens=1
        )
        if "x-ratelimit-limit-tokens" not in response.headers:
            raise RuntimeError("Failed to get dummy response header")
        return response.headers

    @staticmethod
    def _count_prompt_token_capacity(prompt: OAIBasePrompt, **kwargs) -> int:
        max_tokens = kwargs.get("max_tokens", 15)
        n = kwargs.get("n", 1)
        completion_tokens = n * max_tokens

        tokenizer = tiktoken.get_encoding("cl100k_base")
        if isinstance(prompt, str):
            prompt_tokens = len(tokenizer.encode(prompt))
            return prompt_tokens + completion_tokens
        else:
            prompt_tokens = sum(len(tokenizer.encode(p)) for p in prompt)
            return prompt_tokens + completion_tokens

    async def _make_api_call(
        self, prompt: OAIBasePrompt, model_id, start_time, **params
    ) -> list[LLMResponse]:
        LOGGER.debug(f"Making {model_id} call with {self.organization}")
        api_start = time.time()
        api_response: Completion = await self.client.completions.create(
            prompt=prompt, model=model_id, **params
        )
        api_duration = time.time() - api_start
        duration = time.time() - start_time
        cost = estimate_cost(
            model_id,
            api_response.usage.prompt_tokens,
            api_response.usage.completion_tokens,
        )
        return [
            LLMResponse(
                model_id=model_id,
                completion=choice.text,
                stop_reason=choice.finish_reason,
                api_duration=api_duration,
                duration=duration,
                cost=None if cost is None else cost / len(api_response.choices),
                logprobs=choice.logprobs.top_logprobs
                if choice.logprobs is not None
                else None,
            )
            for choice in api_response.choices
        ]

    @staticmethod
    def _print_prompt_and_response(prompt: OAIBasePrompt, responses: list[LLMResponse]):
        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        responses_per_prompt = len(responses) // len(prompt_list)
        responses_list = [
            responses[i : i + responses_per_prompt]
            for i in range(0, len(responses), responses_per_prompt)
        ]
        for i, (prompt, response_list) in enumerate(zip(prompt_list, responses_list)):
            if len(prompt_list) > 1:
                cprint(f"==PROMPT {i + 1}", "white")
            if len(response_list) == 1:
                cprint(f"=={response_list[0].model_id}", "white")
                cprint(prompt, PRINT_COLORS["user"], end="")
                cprint(
                    response_list[0].completion,
                    PRINT_COLORS["assistant"],
                    attrs=["bold"],
                )
            else:
                cprint(prompt, PRINT_COLORS["user"])
                for j, response in enumerate(response_list):
                    cprint(f"==RESPONSE {j + 1} ({response.model_id}):", "white")
                    cprint(
                        response.completion, PRINT_COLORS["assistant"], attrs=["bold"]
                    )
            print()
