"""A tiny in-memory stock list for a shop."""


class Inventory:
    def __init__(self):
        # sku -> {"name": str, "qty": int, "price": float}
        self._items = {}

    def add(self, sku, name, qty, price):
        """Add qty units of an item. A new sku needs name and price; an existing one keeps its own."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        item = self._items.get(sku)
        if item is None:
            self._items[sku] = {"name": name, "qty": qty, "price": float(price)}
        else:
            item["qty"] += qty

    def remove(self, sku, qty):
        self._items[sku]["qty"] -= qty

    def quantity(self, sku):
        return self._items[sku]["qty"]

    def total_value(self):
        return round(sum(i["qty"] * i["price"] for i in self._items.values()), 2)
