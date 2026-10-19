"""Distinguish exhausted billing quota from temporary rate limits."""


def request_id_from_error(error: Exception):
    """Keep provider correlation IDs even when a retry wrapper chains the error."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        request_id = getattr(error, "request_id", None)
        if request_id:
            return request_id
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        request_id = headers.get("request-id") or headers.get("x-request-id")
        if request_id:
            return request_id
        error = error.__cause__ or error.__context__
    return None


def is_credit_exhaustion(error: Exception) -> bool:
    terminal_codes = {"insufficient_quota", "credit_balance_exhausted"}
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict):
            if any(details.get(key) in terminal_codes for key in ("code", "type")):
                return True
    if any(getattr(error, key, None) in terminal_codes for key in ("code", "type")):
        return True
    # Some SDK/provider versions expose only the human-readable billing message.
    message = str(error).lower()
    return any(
        text in message
        for text in (
            "no credits remaining",
            "insufficient credits",
            "credit balance is too low",
        )
    )
