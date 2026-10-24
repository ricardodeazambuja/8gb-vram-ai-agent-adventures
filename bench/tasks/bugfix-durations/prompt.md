Two bug reports came in for `durations.py`:

1. `parse_duration("2h5m")` returns 25200 instead of 7500.
2. `format_duration(3600)` returns "60m" instead of "1h".

There may be more bugs. Fix `durations.py` so that both functions behave exactly as their
docstrings describe. Keep the function names and signatures. `test_durations.py` has the two
reported cases; add your own checks as needed.
