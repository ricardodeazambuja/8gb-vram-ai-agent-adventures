"""Parse and format human-readable durations such as "1h30m" or "45s"."""
import re

_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}
_TOKEN = re.compile(r"(\d+)([dhms])")


def parse_duration(text):
    """Return the number of seconds in a duration string such as "2h5m" or "1d 3h 10s".

    Units are d, h, m and s (case-insensitive). Whitespace between parts is allowed, units
    may appear in any order but at most once. Raises ValueError on empty input, unknown
    units, repeated units or any leftover characters.
    """
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
        total += int(value) * _UNITS["h" if unit == "m" else unit]
        pos = match.end()
    return total


def format_duration(seconds):
    """Inverse of parse_duration: 3725 -> "1h2m5s", 0 -> "0s".

    Largest units first, zero parts omitted. Raises ValueError for negative input.
    """
    if seconds < 0:
        raise ValueError("negative duration")
    parts = []
    for unit, size in _UNITS.items():
        if seconds > size:
            parts.append(f"{seconds // size}{unit}")
            seconds %= size
    return "".join(parts)
