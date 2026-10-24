import csv

import pytest

from inventory import Inventory


def make():
    inv = Inventory()
    inv.add("B2", "Bolt, steel", 50, 0.1)
    inv.add("A1", 'Anchor "heavy"', 3, 12.5)
    inv.add("C3", "Café mug", 7, 4.25)
    inv.add("D4", "Drill", 3, 89.0)
    return inv


def test_existing_behaviour_kept():
    inv = make()
    inv.add("B2", "ignored", 5, 999)
    assert inv.quantity("B2") == 55
    assert inv.total_value() == round(55 * 0.1 + 3 * 12.5 + 7 * 4.25 + 3 * 89.0, 2)


def test_remove_ok_and_zero_stays():
    inv = make()
    inv.remove("A1", 3)
    assert inv.quantity("A1") == 0
    inv.remove("B2", 10)
    assert inv.quantity("B2") == 40


@pytest.mark.parametrize("sku,qty,exc", [
    ("ZZ", 1, KeyError), ("A1", 4, ValueError), ("A1", 0, ValueError), ("A1", -2, ValueError),
])
def test_remove_errors(sku, qty, exc):
    inv = make()
    with pytest.raises(exc):
        inv.remove(sku, qty)
    if sku == "A1":
        assert inv.quantity("A1") == 3


def test_restock_report():
    inv = make()
    assert inv.restock_report(8) == [("A1", 3), ("D4", 3), ("C3", 7)]
    assert inv.restock_report(3) == []
    inv.remove("A1", 3)
    assert inv.restock_report(4) == [("A1", 0), ("D4", 3)]


def test_csv_roundtrip(tmp_path):
    inv = make()
    path = tmp_path / "inv.csv"
    inv.to_csv(path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["sku", "name", "qty", "price"]
    assert len(rows) == 5
    back = Inventory.from_csv(path)
    for sku in ("A1", "B2", "C3", "D4"):
        assert back.quantity(sku) == inv.quantity(sku)
    assert back.total_value() == inv.total_value()
    assert back.restock_report(100) == inv.restock_report(100)
    assert {r[1] for r in rows[1:]} == {"Bolt, steel", 'Anchor "heavy"', "Café mug", "Drill"}
