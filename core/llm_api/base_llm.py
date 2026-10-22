import logging
from enum import Enum, auto
from typing import Dict, List, Optional, Protocol

import attrs
import numpy as np
from pydantic import BaseModel, root_validator

# Delimiters used by Anthropic's legacy completions prompt format.
AI_PROMPT = "\n\nAssistant:"
HUMAN_PROMPT = "\n\nHuman:"

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green"}
LOGGER = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    template: Optional[str] = None

    @root_validator(pre=True)
    def template_defaults(cls, values):
        if values.get("template"):
            from core.prompts import load_template

            defaults = load_template(values["template"])
            return {
                **defaults,
                **values,
                "vars": {**defaults.get("vars", {}), **values.get("vars", {})},
            }
        return values

    partials: Dict[str, str] = {}
    word_limit: Optional[int] = 100
    messages: List[Dict[str, str]] = []
    messages1: List[Dict[str, str]] = []
    messages2: List[Dict[str, str]] = []
    vars: Dict[str, str] = {}


class LanguageModelConfig(BaseModel):
    model: str

    temperature: float = 0.2
    top_p: float = 1.0
    max_tokens: Optional[int] = None
    max_words: int = 10000
    min_words: int = 0
    num_candidates_per_completion: int = 1
    timeout: int = 120


class StopReason(Enum):
    # SPECIAL TREATMENT: providers report different finish-reason vocabularies.
    MAX_TOKENS = auto()
    STOP_SEQUENCE = auto()
    REFUSAL = auto()
    TOOL_CALL = auto()

    @classmethod
    def factory(cls, stop_reason: str) -> "StopReason":
        """
        Parses the openai and anthropic stop reasons into a StopReason enum.
        """
        if stop_reason in ("tool_calls", "tool_use"):
            return cls.TOOL_CALL
        if stop_reason in ["max_tokens", "length"]:
            return cls.MAX_TOKENS
        elif stop_reason in ["stop_sequence", "stop", "end_turn", "completed"]:
            return cls.STOP_SEQUENCE
        elif stop_reason in ["refusal", "content_filter"]:
            return cls.REFUSAL
        raise ValueError(f"Invalid stop reason: {stop_reason}")

    def __repr__(self):
        return self.name


@attrs.frozen()
class LLMResponse:
    model_id: str
    completion: str
    stop_reason: StopReason = attrs.field(converter=StopReason.factory)
    cost: Optional[float]
    duration: Optional[float] = None
    api_duration: Optional[float] = None
    logprobs: Optional[list[dict[str, float]]] = None

    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    tool_calls: list[dict] = attrs.field(factory=list)
    response_items: Optional[list[dict]] = None
    request_id: Optional[str] = None

    def to_dict(self, *, include_opaque=False):
        from core.llm_api.redaction import redact_opaque_data

        data = {
            "model_id": self.model_id,
            "completion": self.completion,
            "stop_reason": self.stop_reason.__repr__(),  # Convert to some JSON-serializable format.
            "duration": self.duration,
            "api_duration": self.api_duration,
            "cost": self.cost,
            "logprobs": self.logprobs,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_calls": self.tool_calls,
            "response_items": self.response_items,
            "request_id": self.request_id,
        }
        # Encrypted reasoning is required for provider replay, but unreadable in
        # results and logs. Only the functional turn cache opts into saving it.
        return data if include_opaque else redact_opaque_data(data)


class ModelAPIProtocol(Protocol):
    async def __call__(
        self,
        model_ids: list[str],
        prompt,
        print_prompt_and_response: bool,
        max_attempts: int,
        **kwargs,
    ) -> list[LLMResponse]:
        raise NotImplementedError


def messages_to_single_prompt(messages) -> str:
    if (
        len(messages) >= 2
        and messages[0]["role"] == "system"
        and messages[1]["role"] == "user"
    ):
        combined_content = messages[0]["content"] + " " + messages[1]["content"]
        messages = [{"role": "user", "content": combined_content}] + messages[2:]
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        tag = AI_PROMPT if role == "assistant" else HUMAN_PROMPT
        prompt += f"{tag} {content}"
    if tag != AI_PROMPT:
        prompt += f"{AI_PROMPT}"
    return prompt.strip()


def convert_to_prob(log_prob: dict, tokens: list) -> tuple[float, float, float]:
    logit1 = log_prob.get(tokens[0], None)
    logit2 = log_prob.get(tokens[1], None)

    if logit1 is None:
        rating = -100
        LOGGER.warning(
            f"Missing token0 {tokens[0]} in log_prob, setting rating to -100.0"
        )
    else:
        rating = logit1

    if logit1 is None:
        logit1 = -100
    if logit2 is None:
        logit2 = -100

    return rating, logit1, logit2


def add_assistant_message(messages: list[dict], assistant_message: str):
    last_role = messages[-1]["role"]
    if last_role == "assistant":
        messages[-1]["content"] += assistant_message
    else:
        messages.append({"role": "assistant", "content": assistant_message})
    return messages
