"""SPECIAL TREATMENT: Claude Messages separates system prompts and text blocks.

Legacy Completions is retained only for reproducing historical configurations.
Model/version choices and compatibility flags live in config/models.yaml.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import attrs
from anthropic import AsyncAnthropic

from core.llm_api.base_llm import (
    LLMResponse,
    ModelAPIProtocol,
    messages_to_single_prompt,
)
from core.llm_api.errors import is_credit_exhaustion
from core.llm_api.registry import estimate_cost, resolve_model
from core.llm_api.tool_calls import anthropic_messages

LOGGER = logging.getLogger(__name__)


@attrs.define()
class AnthropicChatModel(ModelAPIProtocol):
    num_threads: int
    print_prompt_and_response: bool = False
    client: AsyncAnthropic = attrs.field(
        init=False, factory=lambda: AsyncAnthropic(max_retries=0)
    )
    available_requests: asyncio.BoundedSemaphore = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.available_requests = asyncio.BoundedSemaphore(int(self.num_threads))

    @staticmethod
    def messages_request(prompt, spec, params):
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [dict(m) for m in prompt]
        )
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        messages = anthropic_messages([m for m in messages if m["role"] != "system"])
        if not messages:
            raise ValueError("Claude requires at least one non-system message")
        if (
            not spec.get("assistant_prefill", True)
            and messages[-1]["role"] == "assistant"
        ):
            raise ValueError(
                "This model does not support assistant prefill. Move the output-format instruction into the user prompt."
            )
        request = {**spec.get("request_defaults", {}), **params}
        if "tools" in request:
            request["tools"] = [
                dict(
                    name=t["function"]["name"],
                    description=t["function"]["description"],
                    input_schema=t["function"]["parameters"],
                )
                for t in request["tools"]
            ]
        if request.get("tool_choice") == "none":
            request["tool_choice"] = {"type": "none"}
        request["max_tokens"] = request.get("max_tokens") or spec["default_max_tokens"]
        if not spec.get("sampling", False):
            request.pop("temperature", None)
            request.pop("top_p", None)
        elif "temperature" in request and "top_p" in request:
            # Claude versions may reject supplying both sampling controls.
            request.pop("top_p")
        if request.pop("logprobs", None) is not None:
            raise ValueError("Claude does not expose token logprobs")
        return dict(
            model=spec["id"],
            messages=messages,
            **({"system": system} if system else {}),
            **request,
        )

    async def __call__(
        self, model_ids, prompt, print_prompt_and_response, max_attempts, **kwargs
    ):
        if len(model_ids) != 1:
            raise ValueError("Anthropic calls require one model")
        spec = resolve_model(model_ids[0])
        start = time.time()
        if spec["endpoint"] == "messages":
            request = self.messages_request(prompt, spec, kwargs)
        else:
            request = {**spec.get("request_defaults", {}), **kwargs}
            request["max_tokens_to_sample"] = (
                request.pop("max_tokens", None) or spec["default_max_tokens"]
            )
            request.update(
                model=spec["id"],
                prompt=messages_to_single_prompt(prompt)
                if isinstance(prompt, list)
                else prompt,
            )
        response = None
        for attempt in range(max_attempts):
            try:
                async with self.available_requests:
                    api_start = time.time()
                    if spec["endpoint"] == "messages":
                        response = await self.client.messages.create(**request)
                    else:
                        response = await self.client.completions.create(**request)
                    api_duration = time.time() - api_start
                break
            except Exception as error:
                if is_credit_exhaustion(error):
                    raise
                if isinstance(error, (ValueError, TypeError)) or getattr(
                    error, "status_code", None
                ) in (400, 401, 403, 404, 422):
                    raise
                if attempt + 1 == max_attempts:
                    raise RuntimeError(
                        f"Anthropic request failed after {max_attempts} attempts"
                    ) from error
                LOGGER.warning("Anthropic request failed; retrying: %s", error)
                await asyncio.sleep(1.5**attempt)
        if response is None:
            raise RuntimeError("No API attempts were made")
        if spec["endpoint"] == "messages":
            completion = "".join(
                block.text for block in response.content if block.type == "text"
            )
            cost = estimate_cost(
                spec["id"], response.usage.input_tokens, response.usage.output_tokens
            )
        else:
            completion = response.completion
            cost = None  # Legacy API lacks authoritative token usage.
        result = LLMResponse(
            model_id=spec["id"],
            completion=completion,
            input_tokens=response.usage.input_tokens
            if spec["endpoint"] == "messages"
            else None,
            output_tokens=response.usage.output_tokens
            if spec["endpoint"] == "messages"
            else None,
            response_items=[
                block.model_dump(exclude_none=True) for block in response.content
            ]
            if spec["endpoint"] == "messages"
            else None,
            tool_calls=[
                dict(
                    id=block.id,
                    type="function",
                    function=dict(name=block.name, arguments=json.dumps(block.input)),
                )
                for block in response.content
                if block.type == "tool_use"
            ]
            if spec["endpoint"] == "messages"
            else [],
            stop_reason=response.stop_reason,
            cost=cost,
            duration=time.time() - start,
            api_duration=api_duration,
        )
        history = Path("prompt_history")
        history.mkdir(exist_ok=True)
        (history / f"{uuid4()}_prompt.txt").write_text(
            json.dumps({"request": request, "response": result.to_dict()}, indent=2)
        )
        if self.print_prompt_and_response or print_prompt_and_response:
            print(json.dumps(prompt, indent=2), completion)
        return [result]
