import pytest

from durations import format_duration, parse_duration


@pytest.mark.parametrize("text,expected", [
    ("2h5m", 7500), ("45s", 45), ("1d", 86400), ("1d 3h 10s", 97210),
    ("10m 1h", 4200), ("1H30M", 5400), ("  7s  ", 7), ("0s", 0),
])
def test_parse(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("bad", ["", "   ", "5x", "1h1h", "1h30x", "abc", "1h 2", "h"])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (1, "1s"), (59, "59s"), (60, "1m"), (3600, "1h"), (3725, "1h2m5s"),
    (86400, "1d"), (90061, "1d1h1m1s"), (7200, "2h"),
])
def test_format(seconds, expected):
    assert format_duration(seconds) == expected


def test_format_negative():
    with pytest.raises(ValueError):
        format_duration(-1)


def test_roundtrip():
    for s in list(range(0, 4000, 7)) + [86399, 86400, 86401, 200000]:
        assert parse_duration(format_duration(s)) == s
