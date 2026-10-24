"""Parse and format human-readable durations such as "1h30m" or "45s"."""
import re

_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_TOKEN = re.compile(r"(\d+)([dhms])")


def parse_duration(text):
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    total = 0
    seen = set()
    pos = 0
    for match in _TOKEN.finditer(text):
        if text[pos:match.start()].strip():
            raise ValueError(f"unexpected text: {text[pos:match.start()]!r}")
        value, unit = match.groups()
        if unit in seen:
            raise ValueError(f"repeated unit: {unit}")
        seen.add(unit)
        total += int(value) * _UNITS[unit]
        pos = match.end()
    if text[pos:].strip() or not seen:
        raise ValueError(f"unexpected text: {text[pos:]!r}")
    return total


def format_duration(seconds):
    if seconds < 0:
        raise ValueError("negative duration")
    parts = []
    for unit, size in _UNITS.items():
        if seconds >= size:
            parts.append(f"{seconds // size}{unit}")
            seconds %= size
    return "".join(parts) or "0s"
