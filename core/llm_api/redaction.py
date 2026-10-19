"""Keep opaque provider replay state out of human-readable diagnostics."""

REDACTED = "[opaque provider data omitted]"


def redact_opaque_data(value):
    """Return a sanitized copy without mutating live requests or replay caches."""
    if isinstance(value, list):
        return [redact_opaque_data(item) for item in value]
    if not isinstance(value, dict):
        return value
    block_type = value.get("type")
    opaque_keys = {"signature", "encrypted_content"}
    if block_type in ("redacted_thinking", "reasoning.encrypted"):
        opaque_keys.add("data")
    return {
        key: REDACTED if key in opaque_keys and item else redact_opaque_data(item)
        for key, item in value.items()
    }
