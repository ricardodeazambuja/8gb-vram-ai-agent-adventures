import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REF = json.load(open(os.path.join(HERE, "reference.json"), encoding="utf-8"))


def got():
    return json.load(open(os.path.join(HERE, "authors.json"), encoding="utf-8"))


def test_is_object():
    assert isinstance(got(), dict)


def test_authors():
    assert sorted(k.strip() for k in got()) == sorted(REF)


def test_links():
    g = {k.strip(): v.strip() for k, v in got().items()}
    for author, url in REF.items():
        assert g.get(author) == url, author
