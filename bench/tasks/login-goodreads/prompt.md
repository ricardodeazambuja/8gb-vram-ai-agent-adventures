On https://quotes.toscrape.com (a scraping-practice site), each quote shows a "(Goodreads page)"
link to its author, but only after you log in. The login form at
https://quotes.toscrape.com/login accepts any username and password.

Log in, then collect the Goodreads link of every author who appears on **pages 1 and 2** of
https://quotes.toscrape.com. Save the result to `authors.json` as a JSON object mapping each
author's name, exactly as shown on the site, to the link's URL exactly as it appears in the
page, for example:

    {"Albert Einstein": "http://goodreads.com/author/show/9810.Albert_Einstein", ...}

Leave your script in this folder, and make sure `authors.json` exists when you finish.
