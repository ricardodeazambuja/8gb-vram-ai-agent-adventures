from durations import format_duration, parse_duration


def test_reported_parse():
    assert parse_duration("2h5m") == 7500


def test_reported_format():
    assert format_duration(3600) == "1h"
