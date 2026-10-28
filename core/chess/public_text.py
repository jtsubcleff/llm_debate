"""Public text extraction shared by chess solving and interaction protocols."""

import re


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
