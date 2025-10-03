"""Helpers for normalizing trait payloads returned by LLMs."""

from typing import Any, Iterable, List, Set


def normalize_traits_list(payload: Any) -> List[str]:
    """Convert heterogeneous trait payloads into a deduplicated list of strings."""
    normalized: List[str] = []
    seen: Set[str] = set()

    def _append(value: str) -> None:
        stripped = value.strip()
        if not stripped:
            return
        lowered = stripped.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        normalized.append(stripped)

    if not payload:
        return normalized

    if isinstance(payload, str):
        _append(payload)
        return normalized

    if isinstance(payload, dict):
        payload = [payload]

    if isinstance(payload, Iterable):
        for item in payload:
            if isinstance(item, str):
                _append(item)
            elif isinstance(item, dict):
                preferred_keys = ("trait", "name", "value", "label")
                extracted = False
                for key in preferred_keys:
                    value = item.get(key)
                    if isinstance(value, str):
                        _append(value)
                        extracted = True
                if not extracted:
                    for value in item.values():
                        if isinstance(value, str):
                            _append(value)
                            extracted = True
                if not extracted and item:
                    _append(str(item))
            elif item is not None:
                _append(str(item))
        return normalized

    _append(str(payload))
    return normalized


def traits_to_tokens(payload: Iterable[Any]) -> Set[str]:
    """Return lowercase tokens for the provided trait payload."""
    return {value.lower() for value in normalize_traits_list(payload)}

