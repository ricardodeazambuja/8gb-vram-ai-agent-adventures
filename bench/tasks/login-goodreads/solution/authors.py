import json
import requests
from bs4 import BeautifulSoup

s = requests.Session()
tok = BeautifulSoup(s.get("https://quotes.toscrape.com/login", timeout=30).text, "html.parser").find("input", {"name": "csrf_token"})["value"]
s.post("https://quotes.toscrape.com/login", data={"csrf_token": tok, "username": "x", "password": "x"}, timeout=30)
out = {}
for n in (1, 2):
    soup = BeautifulSoup(s.get(f"https://quotes.toscrape.com/page/{n}/", timeout=30).text, "html.parser")
    for q in soup.select(".quote"):
        out[q.select_one(".author").text] = q.find("a", string="(Goodreads page)")["href"]
json.dump(out, open("authors.json", "w"), ensure_ascii=False, indent=1)
