"""Translate the shared chat/tool history to provider-native request items."""


def responses_input(messages):
    items = []
    for message in messages:
        if message.get("response_items") is not None:
            items.extend(message["response_items"])
        elif message["role"] == "tool":
            items.append(
                dict(
                    type="function_call_output",
                    call_id=message["tool_call_id"],
                    output=message["content"],
                )
            )
        else:
            items.append(message)
    return items


def anthropic_messages(messages):
    result = []
    for message in messages:
        if message["role"] == "tool":
            block = dict(
                type="tool_result",
                tool_use_id=message["tool_call_id"],
                content=message["content"],
            )
            if (
                result
                and result[-1]["role"] == "user"
                and isinstance(result[-1]["content"], list)
            ):
                result[-1]["content"].append(block)
            else:
                result.append(dict(role="user", content=[block]))
        elif message.get("response_items") is not None:
            result.append(dict(role="assistant", content=message["response_items"]))
        else:
            result.append(dict(role=message["role"], content=message["content"]))
    return result


def assistant_message(response):
    message = dict(role="assistant", content=response.completion)
    if response.tool_calls:
        message["tool_calls"] = response.tool_calls
    if response.response_items is not None:
        message["response_items"] = response.response_items
    return message
