import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = {b["title"]: b for b in json.load(open(os.path.join(HERE, "reference.json"), encoding="utf-8"))}


def rows():
    with open(os.path.join(HERE, "travel.csv"), newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_script_exists():
    assert os.path.exists(os.path.join(HERE, "scrape_travel.py"))


def test_header():
    assert [c.strip() for c in rows()[0]] == ["title", "price", "rating", "available", "upc"]


def test_titles():
    assert sorted(r[0].strip() for r in rows()[1:]) == sorted(REF)


def test_price_rating():
    assert len(rows()) - 1 == len(REF)
    for r in rows()[1:]:
        b = REF[r[0].strip()]
        assert abs(float(r[1]) - b["price"]) < 1e-6, r
        assert int(r[2]) == b["rating"], r


def test_available_upc():
    assert len(rows()) - 1 == len(REF)
    for r in rows()[1:]:
        b = REF[r[0].strip()]
        assert int(r[3]) == b["available"], r
        assert r[4].strip() == b["upc"], r
