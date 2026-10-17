"""Distinguish exhausted billing quota from temporary rate limits."""


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
