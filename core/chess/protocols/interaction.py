"""Sequential proposer/critic debate and proposer/questioner consultancy.

Each agent turn reuses Direct QA's native tool loop and caches. Only public
text crosses agent boundaries; tool traces and private reasoning stay local.
"""

import json
import re
from pathlib import Path

from core.chess.protocols.direct_qa import (
    prompt_template_name,
    run_chess_turns,
    score_response,
)
from core.chess.tools import TOOL_COUNT_FIELDS
from core.prompts import load_template, render


def turn_schedule(mode, num_rounds):
    if mode not in ("debate", "consultancy"):
        raise ValueError(f"Unknown interaction mode: {mode}")
    if (
        isinstance(num_rounds, bool)
        or not isinstance(num_rounds, int)
        or num_rounds < 1
    ):
        raise ValueError("num_rounds must be a positive integer")
    schedule = []
    for round_index in range(num_rounds):
        schedule.append((round_index + 1, "proposer"))
        if mode == "debate" or round_index < num_rounds - 1:
            schedule.append(
                (round_index + 1, "critic" if mode == "debate" else "questioner")
            )
    return schedule


def turn_cache(cache_dir, index, role):
    return Path(cache_dir) / f"{index:03d}_{role}"


def response_events(response):
    """Use native block order when available (text can surround tool calls)."""
    events = []
    for block in response.get("response_items") or []:
        kind = block.get("type")
        if kind in ("text", "output_text"):
            events.append(("text", block.get("text", "")))
        elif kind == "message":
            events.extend(
                ("text", part.get("text", ""))
                for part in block.get("content", [])
                if part.get("type") in ("text", "output_text")
            )
        elif kind in ("tool_use", "function_call"):
            events.append(("call", block.get("call_id", block.get("id"))))
        # Native thinking/reasoning blocks are always private.
    calls = response.get("tool_calls", [])
    if "".join(value for kind, value in events if kind == "text") == response[
        "completion"
    ] and [value for kind, value in events if kind == "call"] == [
        call["id"] for call in calls
    ]:
        return events
    # Chat APIs provide one text field followed by structured calls.
    return [("text", response["completion"])] + [("call", call["id"]) for call in calls]


def public_turn(trace):
    """Extract public text only; tools and thinking never cross agent boundaries."""
    depth, public = 0, []
    final_text = ""
    for entry in trace:
        if "completion" in entry:
            final_parts = []
            for kind, value in response_events(entry):
                if kind == "call":
                    continue
                parts = re.split(r"(<\s*/?\s*thinking\s*>)", value, flags=re.I)
                for part in parts:
                    if re.fullmatch(r"<\s*thinking\s*>", part, re.I):
                        depth += 1
                    elif re.fullmatch(r"<\s*/\s*thinking\s*>", part, re.I):
                        depth = max(0, depth - 1)
                    elif depth == 0 and part:
                        final_parts.append(part)
                        public.append(dict(type="text", text=part))
            final_text = "".join(final_parts)
    return public, final_text


def format_transcript(transcript):
    """Render text only, including when reading older transcripts with tool events."""
    sections = []
    for turn in transcript:
        events = []
        for event in turn["events"]:
            if event["type"] == "text":
                events.append(event["text"])
        sections.append(
            f"Round {turn['round']} — {turn['role']} ({turn['model']}):\n"
            + "\n".join(events)
        )
    return "\n\n".join(sections)


def score_proposal(puzzle, trace):
    """Score only the opening proposer's final public answer."""
    _, text = public_turn(trace)
    move, correct = score_response(puzzle, text)
    responses = [entry for entry in trace if "completion" in entry]
    normal_stop = responses and responses[-1].get("stop_reason") in (
        "STOP_SEQUENCE",
        "stop",
        "end_turn",
        "completed",
    )
    return move, bool(correct and normal_stop)


def interaction_messages(mode, role, puzzle, transcript, move, use_tools):
    opening = not transcript
    template = load_template(prompt_template_name(use_tools))
    messages = [
        dict(role=m["role"], content=render(m["content"], {"FEN": puzzle.FEN}))
        for m in template["messages"]
    ]
    instructions = load_template("chess_interaction")
    # Keep the existing tool descriptions, but the SAN output contract applies
    # only to the opening proposer. Later turns are arguments or questions.
    base = messages[0]["content"].split("Finish your response")[0]
    messages[0]["content"] = base + "\n" + instructions["privacy"] + "\n"
    key = "opening_" + mode if opening else role + "_" + mode
    messages[0]["content"] += instructions[key]
    if opening:
        messages[1]["content"] += "\n" + instructions["opening_answer"]
    else:
        request = "questions." if role == "questioner" else "argument."
        messages[1]["content"] = (
            f"Original puzzle FEN: {puzzle.FEN}\nLocked proposed move: {move}\n\n"
            f"Full public transcript:\n{format_transcript(transcript)}\n\n"
            f"Provide your next {request}"
        )
    return messages


async def run_interaction(
    api,
    model,
    puzzle,
    cache_dir,
    mode,
    num_rounds,
    questioner_model=None,
    max_tokens=8192,
    max_tool_rounds=10,
    temperature=0.2,
    use_tools=True,
):
    schedule = turn_schedule(mode, num_rounds)
    if mode == "consultancy" and not questioner_model:
        raise ValueError("Consultancy requires an explicit questioner model")
    transcript, trace, turns = [], [], []
    move, correct = "", False
    for index, (round_number, role) in enumerate(schedule):
        agent_model = questioner_model if role == "questioner" else model
        messages = interaction_messages(mode, role, puzzle, transcript, move, use_tools)
        result = await run_chess_turns(
            api,
            agent_model,
            puzzle,
            turn_cache(cache_dir, index, role),
            messages,
            lambda text: ("", False),
            "Tool budget exhausted. Finish your response now."
            + (" End with only your selected SAN move." if index == 0 else ""),
            max_tokens,
            max_tool_rounds,
            temperature,
            use_tools,
        )
        turns.append(result)
        raw = json.loads(result["tool_trace"])
        events, _ = public_turn(raw)
        transcript.append(
            dict(
                round=round_number,
                role=role,
                model=agent_model,
                complete=result["complete"],
                events=events,
            )
        )
        trace.extend(
            dict(round=round_number, role=role, model=agent_model, **e) for e in raw
        )
        if index == 0 and result["complete"]:
            move, correct = score_proposal(puzzle, raw)
        if not result["complete"]:
            break
    return dict(
        **{
            field: sum(t[field] for t in turns)
            for field in (
                *TOOL_COUNT_FIELDS,
                "input_tokens",
                "output_tokens",
                "api_calls",
            )
        },
        cost=None
        if any(t["cost"] is None for t in turns)
        else sum(t["cost"] for t in turns),
        use_tools=use_tools,
        num_rounds=num_rounds,
        final_move=move,
        expected_move=puzzle.expected_move,
        correct=bool(correct),
        complete=len(turns) == len(schedule) and turns[-1]["complete"],
        error=turns[-1]["error"],
        request_id=turns[-1]["request_id"],
        model_response=format_transcript(transcript),
        transcript=json.dumps(transcript),
        tool_trace=json.dumps(trace),
    )
