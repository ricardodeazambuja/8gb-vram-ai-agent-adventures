"""Which form fields Mirage accepts, tried on the probe's example pictures. Runs only inside sandbox.sh.

    bash bench/cactus/sandbox.sh -I /bench/cactus/mirage_forms.py
"""
import json
import sys

sys.path.insert(0, "/cx")
import cactus

m = cactus.Mirage()
E = "/bench/visual-control/examples/"
PCT = {"type": "integer", "minimum": 0, "maximum": 100}
TRIES = [
    ("signal.png", {"traffic_light_color": {"enum": ["red", "yellow", "green"]},
                    "person_on_crosswalk": {"enum": ["yes", "no"]}}),
    ("gauge.png", {"gauge_reading": PCT}),
    ("gauge.png", {"pressure": dict(PCT, description="value the red needle points at")}),
    ("tanks.png", {f"tank_{t}_percent_full": PCT for t in "ABC"}),
    ("tanks.png", {"lowest_tank": {"enum": ["A", "B", "C"]}}),
    ("pick.png", {"object": {"enum": ["circle", "square", "triangle"]}}),
    ("click.png", {"n_buttons": {"type": "integer", "minimum": 0, "maximum": 10}}),
    ("click.png", {"button": {"type": "string"}}),
    ("click.png", {"save_button": {"x-source": "objects"}}),
    ("click.png", {"save_button": {"type": "box"}}),
]
for f, form in TRIES:
    try:
        r = m.extract(E + f, form)
    except Exception as e:  # noqa: BLE001 - unsupported field kinds raise RuntimeError
        r = f"ERR {type(e).__name__}: {e}"
    print(f, json.dumps(form)[:120], "\n   ->", json.dumps(r)[:600])
