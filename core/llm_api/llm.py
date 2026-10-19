import asyncio
import logging
import os
from itertools import chain
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import attrs

from core.llm_api.anthropic_llm import AnthropicChatModel
from core.llm_api.base_llm import LLMResponse
from core.llm_api.openai_llm import (
    OpenAIBaseModel,
    OpenAIChatModel,
    OpenRouterChatModel,
)
from core.llm_api.registry import resolve_model

LOGGER = logging.getLogger(__name__)


@attrs.define()
class ModelAPI:
    anthropic_num_threads: int = 5
    openai_fraction_rate_limit: float = attrs.field(
        default=0.99, validator=attrs.validators.lt(1)
    )
    organization: str = "DEFAULT_ORG"
    print_prompt_and_response: bool = False
    openrouter_timeout: float = attrs.field(
        default=600, validator=attrs.validators.gt(0)
    )

    _adapters: dict = attrs.field(init=False, factory=dict)
    running_cost: float = attrs.field(init=False, default=0)
    unknown_cost_responses: int = attrs.field(init=False, default=0)
    model_timings: dict[str, list[float]] = attrs.field(init=False, factory=dict)
    model_wait_times: dict[str, list[float]] = attrs.field(init=False, factory=dict)

    def __attrs_post_init__(self):
        Path("./prompt_history").mkdir(exist_ok=True)

    def _adapter(self, spec):
        key = (spec["provider"], spec["endpoint"])
        if key not in self._adapters:
            if spec["provider"] == "anthropic":
                adapter = AnthropicChatModel(
                    num_threads=self.anthropic_num_threads,
                    print_prompt_and_response=self.print_prompt_and_response,
                )
            elif spec["provider"] == "openrouter":
                api_key = os.environ.get("OPENROUTER_API_KEY")
                secrets_path = Path("SECRETS")
                if secrets_path.exists():
                    from core.utils import load_secrets

                    api_key = load_secrets(secrets_path).get(
                        "OPENROUTER_API_KEY", api_key
                    )
                if not api_key:
                    raise ValueError(
                        "OpenRouter requires OPENROUTER_API_KEY in the environment or SECRETS."
                    )
                adapter = OpenRouterChatModel(
<<<<<<< HEAD
=======
                    request_timeout=self.openrouter_timeout,
>>>>>>> ded8e85 (Add chess direct QA experiment)
                    frac_rate_limit=self.openai_fraction_rate_limit,
                    organization=None,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    print_prompt_and_response=self.print_prompt_and_response,
                )
            else:
                adapter_type = (
                    OpenAIBaseModel
                    if spec["endpoint"] == "completions"
                    else OpenAIChatModel
                )
                organization = os.environ.get("OPENAI_ORG_ID")
                secrets_path = Path("SECRETS")
                if secrets_path.exists():
                    from core.utils import load_secrets

                    organization = load_secrets(secrets_path).get(
                        self.organization or "DEFAULT_ORG", organization
                    )
                adapter = adapter_type(
                    frac_rate_limit=self.openai_fraction_rate_limit,
                    organization=organization or None,
                    print_prompt_and_response=self.print_prompt_and_response,
                )
            self._adapters[key] = adapter
        return self._adapters[key]

    async def close(self):
        for adapter in self._adapters.values():
            await adapter.client.close()

    async def call_single(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> str:
        assert n == 1, f"Expected a single response. {n} responses were requested."
        responses = await self(
            model_ids,
            prompt,
            max_tokens,
            print_prompt_and_response,
            n,
            max_attempts_per_api_call,
            num_candidates_per_completion,
            is_valid,
            insufficient_valids_behaviour,
            **kwargs,
        )
        assert len(responses) == 1, "Expected a single response."
        return responses[0].completion

    async def __call__(
        self,
        model_ids: Union[str, list[str]],
        prompt: Union[list[dict[str, str]], str],
        max_tokens: int,
        print_prompt_and_response: bool = False,
        n: int = 1,
        max_attempts_per_api_call: int = 10,
        num_candidates_per_completion: int = 1,
        is_valid: Callable[[str], bool] = lambda _: True,
        insufficient_valids_behaviour: Literal[
            "error", "continue", "pad_invalids"
        ] = "error",
        **kwargs,
    ) -> list[LLMResponse]:
        """
        Make maximally efficient API requests for the specified model(s) and prompt.

        Args:
            model_ids: The model(s) to call. If multiple models are specified, the output will be sampled from the
                cheapest model that has capacity. All models must be from the same class (e.g. OpenAI Base,
                OpenAI Chat, or Anthropic Chat). Anthropic chat will error if multiple models are passed in.
                Passing in multiple models could speed up the response time if one of the models is overloaded.
            prompt: The prompt to send to the model(s). Type should match what's expected by the model(s).
            max_tokens: The maximum number of tokens to request from the API (argument added to
                standardize the Anthropic and OpenAI APIs, which have different names for this).
            print_prompt_and_response: Whether to print the prompt and response to stdout.
            n: The number of completions to request.
            max_attempts_per_api_call: Passed to the underlying API call. If the API call fails (e.g. because the
                API is overloaded), it will be retried this many times. If still fails, an exception will be raised.
            num_candidates_per_completion: How many candidate completions to generate for each desired completion. n*num_candidates_per_completion completions will be generated, then is_valid is applied as a filter, then the remaining completions are returned up to a maximum of n.
            is_valid: Candiate completions are filtered with this predicate.
            insufficient_valids_behaviour: What should we do if the remaining completions after applying the is_valid filter is shorter than n.
                error: raise an error
                continue: return the valid responses, even if they are fewer than n
                pad_invalids: pad the list with invalid responses up to n
        """

        assert (
            "max_tokens_to_sample" not in kwargs
        ), "max_tokens_to_sample should be passed in as max_tokens."

        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if not model_ids or n < 1 or num_candidates_per_completion < 1:
            raise ValueError(
                "At least one model and positive candidate counts are required."
            )
        specs = [resolve_model(name) for name in model_ids]
        model_ids = [spec["id"] for spec in specs]
        if len({(spec["provider"], spec["endpoint"]) for spec in specs}) != 1:
            raise ValueError("All model ids must use the same provider and endpoint.")
        if kwargs.get("logprobs") is not None and not all(
            spec.get("logprobs", False) for spec in specs
        ):
            raise ValueError(
                "Selected model profile does not support logprobs; set use_logprobs=false."
            )
        model_class = self._adapter(specs[0])
        kwargs["max_tokens"] = (
            max_tokens
            if max_tokens is not None
            else max(spec["default_max_tokens"] for spec in specs)
        )
        num_candidates = num_candidates_per_completion * n
        if all(spec.get("multiple_completions", False) for spec in specs):
            candidate_responses = await model_class(
                model_ids,
                prompt,
                print_prompt_and_response,
                max_attempts_per_api_call,
                n=num_candidates,
                **kwargs,
            )
        else:
            # Responses and Messages APIs return one completion per request.
            candidate_responses = list(
                chain.from_iterable(
                    await asyncio.gather(
                        *[
                            model_class(
                                model_ids,
                                prompt,
                                print_prompt_and_response,
                                max_attempts_per_api_call,
                                **kwargs,
                            )
                            for _ in range(num_candidates)
                        ]
                    )
                )
            )

        self.running_cost += sum(
            response.cost
            for response in candidate_responses
            if response.cost is not None
        )
        self.unknown_cost_responses += sum(
            response.cost is None for response in candidate_responses
        )
        valid_responses = [
            response
            for response in candidate_responses
            if is_valid(response.completion)
        ]
        num_valid = len(valid_responses)
        success_rate = num_valid / num_candidates
        if success_rate < 1:
            LOGGER.info(f"`is_valid` success rate: {success_rate * 100:.2f}%")

        if num_valid < n:
            match insufficient_valids_behaviour:
                case "error":
                    raise RuntimeError(
                        f"Only found {num_valid} valid responses from {num_candidates} candidates."
                    )
                case "continue":
                    responses = valid_responses
                case "pad_invalids":
                    invalid_responses = [
                        response
                        for response in candidate_responses
                        if not is_valid(response.completion)
                    ]
                    invalids_needed = n - num_valid
                    responses = [*valid_responses, *invalid_responses[:invalids_needed]]
                    LOGGER.info(
                        f"Padded {num_valid} valid responses with {invalids_needed} invalid responses to get {len(responses)} total responses"
                    )
        else:
            responses = valid_responses

        for response in responses:
            self.model_timings.setdefault(response.model_id, []).append(
                response.api_duration
            )
            self.model_wait_times.setdefault(response.model_id, []).append(
                response.duration - response.api_duration
            )
        return responses[:n]

    def reset_cost(self):
        self.running_cost = 0
        self.unknown_cost_responses = 0
