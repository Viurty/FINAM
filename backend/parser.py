from datetime import timezone
import requests
from bs4 import BeautifulSoup
import tldextract
import re
from zoneinfo import ZoneInfo
from datetime import datetime
from collections import OrderedDict
import hashlib
from collections import Counter
import spacy
from urllib.parse import urlparse

def to_utc(dt):
    if not dt:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt

def fetch_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers)
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")
        return None

    if response.status_code == 200:
        return response.text
    else:
        print(f"Ошибка {response.status_code} при запросе {url}")
        return None


def get_canonical_url(soup, fallback_url=None):
    canonical_url = soup.find("link", rel="canonical")
    if canonical_url and canonical_url.get("href"):
        return canonical_url["href"]
    else:
        return fallback_url

def get_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"


def normalize_time(raw_time, timezone='Europe/Moscow'):
    if not raw_time:
        return None
    match = re.match(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})', raw_time)
    if match:
        year, month, day, hour, minute = match.groups()
        try:
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute))
            tz = ZoneInfo(timezone)
            dt_tz = dt.replace(tzinfo=tz)
            return dt_tz.isoformat()
        except ValueError:
            pass
    if 'T' in raw_time and ':' in raw_time:
        return raw_time
    return raw_time  # Fallback

def get_publish_time(soup, url=None):
    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        return normalize_time(time_tag["datetime"])

    meta_time = soup.find("meta", attrs={"property": "article:published_time"})
    if meta_time and meta_time.get("content"):
        return normalize_time(meta_time["content"])

    div_date = soup.find("div", class_="article__info-date")
    if div_date:
        a_tag = div_date.find("a")
        if a_tag:
            text = a_tag.get_text(strip=True)
            match = re.search(r'(\d{2}):(\d{2})\s+(\d{2})\.(\d{2})\.(\d{4})', text)
            if match:
                hh, mm, day, month, year = match.groups()
                raw = f"{year}{month}{day}T{hh}{mm}"
                return normalize_time(raw)

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        if "NewsArticle" in content and "datePublished" in content:
            double_match = re.search(r'\\"datePublished\\"\s*:\s*\\"([^\\"]+)\\"', content)
            if double_match:
                return normalize_time(double_match.group(1))

            triple_match = re.search(r'\\\\\\"datePublished\\\\\\"\\s*:\\s*\\\\\\"([^\\\\\\"]+)\\\\\\"', content)
            if triple_match:
                return normalize_time(triple_match.group(1))

            simple_match = re.search(r'"datePublished"\s*:\s*"([^"]+)"', content)
            if simple_match:
                return normalize_time(simple_match.group(1))

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        pub_dt_match = re.search(r'\\"published_dt\\"\s*:\s*"([^"]+)"', content)
        if pub_dt_match:
            return normalize_time(pub_dt_match.group(1))

    if url:
        path = urlparse(url).path
        date_match = re.match(r'.*/(\d{4})/(\d{2})/(\d{2})/.*', path)
        if date_match:
            year, month, day = date_match.groups()
            return normalize_time(f"{year}{month}{day}T0000")
        ria_match = re.match(r'/(\d{4})(\d{2})(\d{2})/.*', path)
        if ria_match:
            year, month, day = ria_match.groups()
            return normalize_time(f"{year}{month}{day}T0000")

    time_tag_text = soup.find("time")
    if time_tag_text:
        return normalize_time(time_tag_text.get_text(strip=True))

    return None


def get_modified_time(soup, url=None):
    meta_time = soup.find("meta", attrs={"property": "article:modified_time"})
    if meta_time and meta_time.get("content"):
        return normalize_time(meta_time["content"])

    span_mod = soup.find("span", class_="article__info-date-modified")
    if span_mod:
        text = span_mod.get_text(strip=True)
        match = re.search(r'обновлено:\s*(\d{2}):(\d{2})\s+(\d{2})\.(\d{2})\.(\d{4})', text)
        if match:
            hh, mm, day, month, year = match.groups()
            raw = f"{year}{month}{day}T{hh}{mm}"
            return normalize_time(raw)

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        if "NewsArticle" in content and "dateModified" in content:
            double_match = re.search(r'\\"dateModified\\"\s*:\s*\\"([^\\"]+)\\"', content)
            if double_match:
                return normalize_time(double_match.group(1))

            triple_match = re.search(r'\\\\\\"dateModified\\\\\\"\\s*:\\s*\\\\\\"([^\\\\\\"]+)\\\\\\"', content)
            if triple_match:
                return normalize_time(triple_match.group(1))

            simple_match = re.search(r'"dateModified"\s*:\s*"([^"]+)"', content)
            if simple_match:
                return normalize_time(simple_match.group(1))

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        upd_dt_match = re.search(r'\\"updated_dt\\"\s*:\s*"([^"]+)"', content)
        if upd_dt_match:
            return normalize_time(upd_dt_match.group(1))

    published = get_publish_time(soup, url)
    if published:
        return published

    if url:
        path = urlparse(url).path
        date_match = re.match(r'.*/(\d{4})/(\d{2})/(\d{2})/.*', path)
        if date_match:
            year, month, day = date_match.groups()
            return normalize_time(f"{year}{month}{day}T0000")
        ria_match = re.match(r'/(\d{4})(\d{2})(\d{2})/.*', path)
        if ria_match:
            year, month, day = ria_match.groups()
            return normalize_time(f"{year}{month}{day}T0000")

    return None

def get_article_title(soup):
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        return title_tag.get_text(strip=True)

    meta_title = soup.find("meta", attrs={"name": "title"})
    if not meta_title:
        meta_title = soup.find("meta", attrs={"property": "og:title"})
    if meta_title and meta_title.get("content"):
        return meta_title["content"]

    h1_tag = soup.find("h1")
    if h1_tag and h1_tag.get_text(strip=True):
        return h1_tag.get_text(strip=True)

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        if "NewsArticle" in content and "headline" in content:
            double_match = re.search(r'\\"headline\\"\s*:\s*\\"([^\\"]+)\\"', content)
            if double_match:
                return double_match.group(1)

            triple_match = re.search(r'\\\\\\"headline\\\\\\"\\s*:\\s*\\\\\\"([^\\\\\\"]+)\\\\\\"', content)
            if triple_match:
                return triple_match.group(1)

            simple_match = re.search(r'"headline"\s*:\s*"([^"]+)"', content)
            if simple_match:
                return simple_match.group(1)

    return soup.find("h1").get_text(strip=True) if soup.find("h1") else None


def get_article_lead(soup):
    lead_tag = soup.find("summary", class_=re.compile(r"PageLead|lead", re.I))
    if not lead_tag:
        lead_tag = soup.find("div", class_=re.compile(r"PageLead|lead", re.I))
    if lead_tag and lead_tag.get_text(strip=True):
        return lead_tag.get_text(strip=True)

    meta_desc = soup.find("meta", attrs={"name": "description"})
    if not meta_desc:
        meta_desc = soup.find("meta", attrs={"property": "og:description"})
    if meta_desc and meta_desc.get("content"):
        return meta_desc["content"]

    for script in soup.find_all("script"):
        content = script.get_text(strip=True)
        if "NewsArticle" in content or "lead" in content:
            double_match = re.search(r'\\"lead\\"\s*:\s*\\"([^\\"]+)\\"', content)
            if double_match:
                return double_match.group(1)

            triple_match = re.search(r'\\\\\\"lead\\\\\\"\\s*:\\s*\\\\\\"([^\\\\\\"]+)\\\\\\"', content)
            if triple_match:
                return triple_match.group(1)

            desc_double = re.search(r'\\"description\\"\s*:\s*\\"([^\\"]+)\\"', content)
            if desc_double:
                return desc_double.group(1)

            desc_triple = re.search(r'\\\\\\"description\\\\\\"\\s*:\\s*\\\\\\"([^\\\\\\"]+)\\\\\\"', content)
            if desc_triple:
                return desc_triple.group(1)

            simple_lead = re.search(r'"lead"\s*:\s*"([^"]+)"', content)
            if simple_lead:
                return simple_lead.group(1)

    first_p = soup.find("article").find("p") if soup.find("article") else None
    if first_p:
        return first_p.get_text(strip=True)

    return None

def get_article_text(soup):
    article_container = (
        soup.find("div", class_=re.compile(r"article__body")) or
        soup.find("div", id="article_text") or
        soup.find("article") or
        soup.find("main") or
        soup.find("body")
    )

    if not article_container:
        return None

    paragraphs = []
    for p in article_container.find_all("p", recursive=True):
        parent_class = " ".join(p.parent.get("class", [])) if p.parent else ""
        if re.search(r"header|footer|ad|sidebar|related|comment", parent_class, re.I):
            continue

        text = p.get_text(strip=True)
        if (
            text and
            len(text) > 20 and
            re.search(r'[А-Яа-яA-Za-z]{3,}', text) and
            not re.match(r'^\s*(Фото|©|Ссылка|Обновлено|Источник):\s*$', text, re.I)
        ):
            paragraphs.append(text)
    unique_paragraphs = list(OrderedDict.fromkeys(paragraphs))

    if len(unique_paragraphs) < 5:
        h1 = article_container.find("h1")
        if h1:
            for sibling in h1.find_all_next("p", limit=30):
                text = sibling.get_text(strip=True)
                if len(text) > 20:
                    unique_paragraphs.append(text)
        unique_paragraphs = list(OrderedDict.fromkeys(unique_paragraphs))

    if unique_paragraphs:
        full_text = '\n\n'.join(unique_paragraphs)
        full_text = re.sub(r'\s{2,}', ' ', full_text)
        return full_text.strip()

    return None

def detect_language(text):
    if not text or len(text.strip()) < 10:
        return 'unknown'

    normalized = re.sub(r'[^\w\s]', ' ', text.lower())
    words = normalized.split()

    cyrillic_chars = len(re.findall(r'[а-яё]', text))
    latin_chars = len(re.findall(r'[a-z]', text))

    if cyrillic_chars > latin_chars * 1.5:
        return 'ru'
    elif latin_chars > cyrillic_chars * 1.5:
        return 'en'

    ru_words = {'это', 'что', 'как', 'в', 'на', 'с', 'для', 'не', 'по', 'из', 'тасс', 'москва', 'октября'}
    en_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'for', 'on', 'with', 'as', 'app', 'store'}

    word_count = Counter(words)
    ru_score = sum(word_count.get(word, 0) for word in ru_words)
    en_score = sum(word_count.get(word, 0) for word in en_words)

    if ru_score > en_score:
        return 'ru'
    elif en_score > ru_score:
        return 'en'

    return 'unknown'


def extract_entities_ru(text: str):
    nlp_ru = spacy.load("ru_core_news_sm")
    doc = nlp_ru(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['ORG']:
            entities.append(ent.text.strip())

    return list(set(entities))


def extract_entities_en(text: str):
    nlp_en = spacy.load("en_core_web_sm")

    doc = nlp_en(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['ORG']:
            entities.append(ent.text.strip())

    return list(set(entities))


def get_lead_by_text(text: str) -> str:
    if not text:
        return ""

    normalized = re.sub(r'\s+', ' ', text.strip())
    parts = re.split(r'(?<=[\.!\?…])\s+', normalized)
    sentences = [p.strip() for p in parts if p.strip()]

    if not sentences:
        return ""

    lead = " ".join(sentences[:2]).strip()
    return lead

def get_data(url):
    html = fetch_html(url)
    if html is None:
        return None
    soup = BeautifulSoup(html, "html.parser")
    canonical_url = get_canonical_url(soup, fallback_url=url)
    domain = get_domain(canonical_url)
    published_time = get_publish_time(soup, url)
    crawl_time = datetime.now().replace(microsecond=0).isoformat()
    last_modified = get_modified_time(soup, url)
    title = get_article_title(soup)
    if published_time is None:
        return None
    if title is None:
        return None
    lead = get_article_lead(soup)
    text = get_article_text(soup)
    if text is None:
        return None
    if lead is None:
        lead = get_lead_by_text(text)
    language = detect_language(text)
    if text:
        hash_object = hashlib.sha256(text.encode('utf-8'))
        content_hash = hash_object.hexdigest()
    else:
        content_hash = None
    if language == 'ru' and text:
        entities = extract_entities_ru(text)
    elif language == 'en' and text:
        entities = extract_entities_en(text)
    else:
        entities = []
    data = {
        "url" :url,
        "canonical_url":canonical_url ,
        "domain" : domain ,
        "published_time" : published_time ,
        "crawl_time" : crawl_time ,
        "last_modified" : last_modified ,
        "title" : title ,
        "lead" : lead ,
        "text" : text ,
        "language" : language ,
        "content_hash" : content_hash ,
        "entities" : entities,
        "tickets" : []
    }

    return data









