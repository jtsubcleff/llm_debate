"""Direct QA with bounded native tool use and resumable per-turn caching."""

import hashlib
import json
from pathlib import Path

from core.chess.results import write_json
from core.chess.tools import VALIDATE_MOVE_SEQUENCE, validate_move_sequence
from core.llm_api.base_llm import LLMResponse, StopReason
from core.llm_api.tool_calls import assistant_message
from core.prompts import load_template, render


def prompt_template_name(use_tools=True):
    return "chess_direct_qa" if use_tools else "chess_direct_qa_no_tools"


def score_response(puzzle, response):
    answer = response.split()[-1] if response.split() else ""
    return answer, answer == puzzle.expected_move


async def run_direct_qa(
    api,
    model,
    puzzle,
    cache_dir,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    use_tools=True,
):
    messages = [
        dict(role=m["role"], content=render(m["content"], {"FEN": puzzle.FEN}))
        for m in load_template(prompt_template_name(use_tools))["messages"]
    ]
    texts, trace = [], []
    input_tokens = output_tokens = 0
    cost = 0.0
    unknown_cost = False
    complete, error, answer, correct = False, "", "", False
    tool_rounds = max_tool_rounds if use_tools else 0
    try:
        for turn in range(tool_rounds + 1):
            cache_key = hashlib.sha256(puzzle.puzzleId.encode()).hexdigest()
            cache = Path(cache_dir) / f"{cache_key}_{turn}.json"
            if cache.exists():
                data = json.loads(cache.read_text())
                # Enum names differ from provider finish reasons.
                data["stop_reason"] = {
                    "MAX_TOKENS": "max_tokens",
                    "STOP_SEQUENCE": "stop",
                    "TOOL_CALL": "tool_calls",
                    "REFUSAL": "refusal",
                }[data["stop_reason"]]
                response = LLMResponse(**data)
            else:
                kwargs = dict(temperature=temperature)
                if use_tools:
                    kwargs["tools"] = [VALIDATE_MOVE_SEQUENCE]
                if use_tools and turn == tool_rounds:
                    # Give a final answer turn after all permitted tool rounds.
                    messages.append(
                        dict(
                            role="user",
                            content="Tool budget exhausted. Return your final SAN move now.",
                        )
                    )
                    kwargs["tool_choice"] = "none"
                response = (
                    await api(model, messages, max_tokens=max_tokens, **kwargs)
                )[0]
                write_json(cache, response.to_dict())
            input_tokens += response.input_tokens or 0
            output_tokens += response.output_tokens or 0
            unknown_cost |= response.cost is None
            cost += response.cost or 0
            texts.append(response.completion)
            trace.append(response.to_dict())
            if not response.tool_calls:
                answer, correct = score_response(puzzle, response.completion)
                complete = True
                # A truncated or refused answer is never a successful solution.
                correct &= response.stop_reason == StopReason.STOP_SEQUENCE
                break
            if not use_tools:
                raise RuntimeError(
                    "Model requested tools while use_tools=false; no tools were executed"
                )
            if turn == tool_rounds:
                raise RuntimeError(
                    "Model requested tools after exhausting the tool budget"
                )
            messages.append(assistant_message(response))
            for call in response.tool_calls:
                try:
                    if call["function"]["name"] != "validate_move_sequence":
                        raise ValueError("Unknown tool")
                    arguments = json.loads(call["function"]["arguments"])
                    result = validate_move_sequence(puzzle.FEN, **arguments)
                except (ValueError, TypeError) as exc:
                    result = dict(error=str(exc))
                trace.append(dict(tool_call=call, result=result))
                messages.append(
                    dict(
                        role="tool", tool_call_id=call["id"], content=json.dumps(result)
                    )
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return dict(
        use_tools=use_tools,
        model_response="\n\n".join(texts),
        final_move=answer,
        expected_move=puzzle.expected_move,
        correct=bool(correct),
        complete=complete,
        error=error,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=None if unknown_cost else cost,
        tool_trace=json.dumps(trace),
        api_calls=sum("completion" in entry for entry in trace),
    )
