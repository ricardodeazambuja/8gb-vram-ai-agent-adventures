Write a command-line tool `wordfreq.py` (Python 3, standard library only) that counts words in a
text file:

    python3 wordfreq.py FILE [--top N] [--min-length L] [--ignore WORD [WORD ...]]

- A word is a run of ASCII letters, which may contain internal apostrophes (`didn't` is one word,
  `'quoted'` is the word `quoted`). Digits and all other characters separate words. Counting is
  case-insensitive; report words in lowercase.
- `--top N` (default 10): print at most N words. `--min-length L` (default 1): skip words shorter
  than L characters. `--ignore`: skip these words (case-insensitive).
- `FILE` may be `-` to read standard input.
- Print a JSON array to standard output: `[{"word": "the", "count": 4}, ...]`, sorted by count
  (highest first), then alphabetically.
- If the file does not exist, print an error to standard error and exit with status 2.

`sample.txt` is a small example input.
