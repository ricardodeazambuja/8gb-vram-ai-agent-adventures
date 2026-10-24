import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, "reference.json"), encoding="utf-8"))


def got():
    return json.load(open(os.path.join(HERE, "quotes.json"), encoding="utf-8"))


def norm(t):
    return t.strip().strip("“”\"").strip()


def test_count():
    assert len(got()) == 30


def test_authors_in_order():
    assert [q["author"].strip() for q in got()] == [q["author"] for q in REF]


def test_texts():
    assert [norm(q["text"]) for q in got()] == [q["text"] for q in REF]


def test_no_curly_marks():
    assert got() and all(not q["text"].strip().startswith("“") for q in got())


def test_tags():
    assert [list(q["tags"]) for q in got()] == [q["tags"] for q in REF]
