import json
from playwright.sync_api import sync_playwright

out = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    for n in (1, 2, 3):
        pg.goto(f"https://quotes.toscrape.com/js/page/{n}/")
        pg.wait_for_selector(".quote")
        for q in pg.query_selector_all(".quote"):
            out.append({"text": q.query_selector(".text").inner_text().strip("“”"),
                        "author": q.query_selector(".author").inner_text(),
                        "tags": [t.inner_text() for t in q.query_selector_all(".tag")]})
    b.close()
json.dump(out, open("quotes.json", "w"), ensure_ascii=False, indent=1)
