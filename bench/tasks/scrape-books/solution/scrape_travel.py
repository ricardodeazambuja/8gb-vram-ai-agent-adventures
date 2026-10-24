import csv, re
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/catalogue/category/books/travel_2/index.html"
RATING = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
soup = BeautifulSoup(requests.get(BASE, timeout=30).content, "html.parser")
rows = []
for pod in soup.select("article.product_pod"):
    a = pod.h3.a
    page = BeautifulSoup(requests.get(urljoin(BASE, a["href"]), timeout=30).content, "html.parser")
    avail = int(re.search(r"\((\d+) available\)", page.select_one(".availability").text).group(1))
    upc = page.find("th", string="UPC").find_next("td").text.strip()
    rating = RATING[[c for c in pod.select_one(".star-rating")["class"] if c != "star-rating"][0]]
    rows.append([a["title"], pod.select_one(".price_color").text.lstrip("£Â"), rating, avail, upc])
with open("travel.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["title", "price", "rating", "available", "upc"]); w.writerows(rows)
