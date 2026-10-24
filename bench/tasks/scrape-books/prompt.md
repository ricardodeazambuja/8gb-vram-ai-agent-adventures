The site https://books.toscrape.com is a sandbox bookshop made for scraping practice.

Write `scrape_travel.py` that collects every book in the **Travel** category and writes
`travel.csv` (UTF-8) with exactly these columns, in this order:

    title,price,rating,available,upc

- `title`: the book's full title (the listing page truncates long titles; use the full one).
- `price`: the price as a plain number, without the currency sign (e.g. `45.17`).
- `rating`: the star rating as an integer from 1 to 5.
- `available`: the number of copies in stock, from the book's own page ("In stock (19 available)" → `19`).
- `upc`: the UPC code from the book's own page.

Run your script so that `travel.csv` is present when you finish.
