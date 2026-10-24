`inventory.py` holds a small `Inventory` class. Extend it as follows, without changing the
behaviour of `add`, `quantity` or `total_value`:

1. `remove(sku, qty)` must raise `KeyError` for an unknown sku, and `ValueError` if `qty` is not
   positive or is more than the quantity in stock. An item whose quantity reaches 0 stays in the
   inventory with quantity 0.
2. Add `restock_report(threshold)`: return a list of `(sku, qty)` tuples for every item whose
   quantity is strictly below `threshold`, sorted by quantity ascending, then by sku.
3. Add `to_csv(path)` and a classmethod `from_csv(path)` that round-trip the inventory through a
   CSV file with the header `sku,name,qty,price`. Names may contain commas, quotes and non-ASCII
   characters. `from_csv` returns a new `Inventory` with the same items, quantities and prices.

Test your work.
