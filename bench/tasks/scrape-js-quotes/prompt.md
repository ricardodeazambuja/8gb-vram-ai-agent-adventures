The page https://quotes.toscrape.com/js/ is a scraping-practice site whose quotes are rendered
by JavaScript (the raw HTML has no quote elements).

Collect the quotes on the **first three pages** of that JavaScript version
(`/js/`, `/js/page/2/`, `/js/page/3/`) and save them to `quotes.json` as a JSON list, in page
order, of objects:

    {"text": ..., "author": ..., "tags": [...]}

`text` is the quote without the surrounding curly quotation marks (“ ”). `tags` keeps the
site's order. Leave your scraping script in this folder as well, and make sure `quotes.json`
exists when you finish.
