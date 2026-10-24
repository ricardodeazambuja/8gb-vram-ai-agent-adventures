import csv


class Inventory:
    def __init__(self):
        self._items = {}

    def add(self, sku, name, qty, price):
        if qty <= 0:
            raise ValueError("qty must be positive")
        item = self._items.get(sku)
        if item is None:
            self._items[sku] = {"name": name, "qty": qty, "price": float(price)}
        else:
            item["qty"] += qty

    def remove(self, sku, qty):
        item = self._items[sku]
        if qty <= 0 or qty > item["qty"]:
            raise ValueError("bad qty")
        item["qty"] -= qty

    def quantity(self, sku):
        return self._items[sku]["qty"]

    def total_value(self):
        return round(sum(i["qty"] * i["price"] for i in self._items.values()), 2)

    def restock_report(self, threshold):
        return sorted(((s, i["qty"]) for s, i in self._items.items() if i["qty"] < threshold),
                      key=lambda x: (x[1], x[0]))

    def to_csv(self, path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["sku", "name", "qty", "price"])
            for s, i in self._items.items():
                w.writerow([s, i["name"], i["qty"], i["price"]])

    @classmethod
    def from_csv(cls, path):
        inv = cls()
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                inv._items[r["sku"]] = {"name": r["name"], "qty": int(r["qty"]), "price": float(r["price"])}
        return inv
