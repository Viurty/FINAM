import os
from urllib.parse import urlparse

import feedparser
from email.utils import parsedate_to_datetime
from datetime import timezone
from datetime import datetime, timedelta
import requests

rss_links = [
    'https://lenta.ru/rss',
    'https://tass.com/rss/v2.xml',
    'https://rssexport.rbc.ru/rbcnews/news/30/full.rss',
    'https://www.kommersant.ru/rss-list',
    'https://rg.ru/xml/index.xml',
    'https://www.kommersant.ru/rss/review.xml',
    'https://www.kommersant.ru/rss/section-economics.xml',
    'https://www.kommersant.ru/rss/section-business.xml',
    'https://www.kommersant.ru/rss/news.xml',
    'http://feeds.foxnews.com/foxnews/latest',
    'https://feeds.a.dj.com/rss/RSSWorldNews.xml',
    'http://rss.cnn.com/rss/edition.rss',
]



def get_links(count=20, rss=None):
    if rss is None:
        rss = rss_links
    all_links = []

    for rss_url in rss:
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries:
                continue

            links = []
            for entry in feed.entries:

                links.append(entry.link)

                if len(links) >= count:
                    break

            all_links.extend(links)

        except Exception as e:
            print(f"Ошибка при обработке {rss_url}: {e}")

    return all_links

def _is_valid_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def build_time_fragment(time_keyword: str, tz: str = "Europe/Moscow"):
    if isinstance(time_keyword, (list, tuple)) and len(time_keyword) == 2:
        start = datetime.fromisoformat(time_keyword[0]).date()
        end = datetime.fromisoformat(time_keyword[1]).date() + timedelta(days=1)
    else:
        return {"q_fragment": "", "start": None, "end": None}

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    return {"q_fragment": f" от {start_s} до {end_s}", "start": start_s, "end": end_s}


def build_query(topic: str, time_keyword, tz: str = "Europe/Moscow"):
    frag = build_time_fragment(time_keyword, tz)
    if frag["q_fragment"]:
        q = f"{topic} {frag['q_fragment']}"
    else:
        q = topic
    return q

def search(query: str, max_pages: int = 1):
    GOOGLE_ENDPOINT = os.getenv("ENDPOINT")
    api_key = os.getenv("API")
    cx = os.getenv("СХ")

    links = []
    seen = set()
    start = 1
    for page in range(max_pages):
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": 10,
            "start": start
        }
        try:
            r = requests.get(GOOGLE_ENDPOINT, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            if not items:
                break
            for it in items:
                url = it.get("link")
                if url and _is_valid_url(url) and url not in seen:
                    seen.add(url)
                    links.append(url)
        except requests.RequestException as e:
            print(f"[ERROR] Ошибка запроса: {e}")
            break

        start += 10

    return links