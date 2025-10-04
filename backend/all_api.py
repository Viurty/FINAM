import os
import requests
from main_content_extractor import MainContentExtractor
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import re
from urllib.parse import urlparse, urlunparse
import hashlib
from transformers import pipeline

# --- Загружаем переменные окружения ---
load_dotenv()
POLYGON_API_KEY = os.getenv("POLYGON_IO_API")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API")
FMP_API_KEY = os.getenv("FINANCIAL_MODELING_PREP_API")
NEWS_API_KEY = os.getenv("NEWS_API")

MAX_TOPIC_LENGTH = 100
try:
    ner_pipeline = pipeline(
        "ner",
        model="Babelscape/wikineural-multilingual-ner",
        aggregation_strategy="simple",
        device=-1  # Принудительно CPU
    )
    print("✅ NER модель загружена успешно")
except Exception as e:
    print(f"❌ Ошибка загрузки NER модели: {e}")
    ner_pipeline = None


# --- Даты с фронтенда ---
start_date_str = "2025-10-03"
end_date_str = "2025-10-04"

# --- Вспомогательные функции ---
def has_duplicate_words(text):
    """Проверяет, содержит ли строка повторяющиеся слова подряд."""
    words = text.split()
    if len(words) < 2:
        return False

    # Проверяем на полное дублирование фразы
    text_lower = text.lower()
    half_len = len(text) // 2
    if len(text) > 3 and text_lower[:half_len] == text_lower[half_len:half_len * 2]:
        return True

    # Проверяем на повторяющиеся слова
    for i in range(len(words) - 1):
        if words[i].lower() == words[i + 1].lower():
            return True

    return False

def remove_nested_duplicates(nested_list):
    seen = set()
    unique_list = []
    for sublist in nested_list:
        # Преобразуем вложенный список в кортеж для хеширования
        tuple_sublist = tuple(sublist)
        if tuple_sublist not in seen:
            seen.add(tuple_sublist)
            unique_list.append(sublist)
    return unique_list

def to_final_format(item: dict) -> dict:
    final = {
        "url": item.get("source_url"),
        "canonical_url": item.get("canonical_url"),
        "domain": item.get("source_domain"),
        "published_time": item.get("published_at_utc") or "",
        "crawl_time": datetime.now().isoformat(),
        "last_modified": item.get("published_at_utc") or "",
        "title": item.get("title"),
        "lead": item.get("summary"),
        "text": item.get("content"),
        "language": item.get("language"),
        "content_hash": item.get("content_hash"),
        "entities": item.get("topics", []),
        "tickers": item.get("tickers", [])
    }
    return final
def generate_content_hash(text: str) -> str:
    if not text:
        return ""
    text = text.strip().encode("utf-8")
    return hashlib.sha256(text).hexdigest()
def get_canonical_url(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url.strip())
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower().lstrip("www.")
    path = parsed.path.rstrip("/")

    canonical = urlunparse((scheme, netloc, path, "", "", ""))
    return canonical

def get_source_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    return parsed.netloc.lower().lstrip("www.")


def extract_entities_universal(text: str):
    """Универсальная многоязычная NER с поддержкой русского + английский."""
    if not text or len(text.strip()) < 10:
        return []

    entities = []

    try:
        # Проверяем, что pipeline инициализирован
        if 'ner_pipeline' not in globals() or ner_pipeline is None:
            print("⚠️ NER pipeline не инициализирован!")
            return []

        # Ограничиваем длину текста (wikineural работает с токенами, не символами)
        text_sample = text[:2000]  # Увеличиваем лимит

        print(f"🔍 Анализ текста длиной {len(text_sample)} символов...")
        ner_results = ner_pipeline(text_sample)

        print(f"📊 Найдено {len(ner_results)} сущностей через NER")

        for entity in ner_results:
            if entity['entity_group'] in ['ORG', 'PER', 'LOC'] and entity['score'] > 0.6:
                clean_entity = entity['word'].strip().replace('▁', ' ').strip()
                if 3 <= len(clean_entity) <= MAX_TOPIC_LENGTH:
                    entities.append(clean_entity)
                    print(f"  ✅ {clean_entity} ({entity['entity_group']}, {entity['score']:.2f})")

    except Exception as e:
        print(f"❌ Критическая ошибка NER: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

    # Regex-паттерны как fallback
    patterns = [
        r'\b([A-ZА-ЯЁ][a-zA-Zа-яёА-ЯЁ&\s]+(?:Inc\.|Corp\.|Ltd\.|LLC|Co\.|Corporation|Limited|Group|Holdings))',
        r'\b(ООО|ПАО|АО|ЗАО|ИП|ГК)\s+[«"]?([А-ЯЁ][а-яёА-ЯЁ\s\-]+)[»"]?',
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text[:2000])
        for match in matches:
            entity = match.group(0).strip()
            if 3 <= len(entity) <= MAX_TOPIC_LENGTH:
                entities.append(entity)

    # Удаление дубликатов
    entities = list(dict.fromkeys(entities))  # Сохраняет порядок
    entities = [e for e in entities if not has_duplicate_words(e)]

    print(f"📋 Итого извлечено: {len(entities)} уникальных сущностей")
    return entities

def parser_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на успешный статус ответа
        response.encoding = 'utf-8'
        content = response.text

        # Попытка извлечь основной контент
        return MainContentExtractor.extract(content, output_format="text")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе URL {url}: {e}")
    except Exception as e:
        print(f"Ошибка при обработке URL {url}: {e}")
    return ""  # Возвращаем пустую строку в случае ошибки

def normalize_date_str(date_str):
    year, month, day = date_str.split("-")
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

def parse_date(date_str):
    formats = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

# --- Polygon.io ---
def get_polygon_news_all(start_date, end_date, fetch_full_text=True):
    news_list = []
    seen_ids = set()  # чтобы отслеживать дубликаты
    start_fixed = normalize_date_str(start_date)
    end_fixed = normalize_date_str(end_date)
    offset = 0
    page_limit = 100
    max_pages = 50  # максимум 5000 новостей (50 * 100)

    for page in range(max_pages):
        print(f"Запрос страницы {page + 1} новостей Polygon.io...")
        url = (
            f"https://api.polygon.io/v2/reference/news?"
            f"published_utc.gte={start_fixed}&published_utc.lte={end_fixed}"
            f"&limit={page_limit}&offset={offset}&apiKey={POLYGON_API_KEY}"
        )

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ Ошибка при запросе Polygon.io: {e}")
            break

        results = data.get("results", [])
        if not results:
            print("Достигнут конец ленты Polygon.io.")
            break

        new_items = 0
        for item in results:
            print("Обработка новости:", item.get("title", "")[:60], " от Polygon.io")
            article_id = item.get("id") or item.get("article_url")
            if article_id in seen_ids:
                continue
            seen_ids.add(article_id)

            content = item.get("content", "")
            if fetch_full_text and (not content or "[+" in content) and len(content) < 200:
                full_text = parser_html(item.get("article_url"))
                if full_text:
                    content = full_text
            if content == "":
                continue
            topics = item.get("keywords", [])
            # topics += extract_entities_en(content)
            # topics = remove_nested_duplicates(topics)
            # topics = [topic for topic in topics if len(topic) <= MAX_TOPIC_LENGTH and not has_duplicate_words(topic)]
            news_list.append(to_final_format({
                "title": item.get("title"),
                "summary": item.get("description"),
                "content": content,
                "author": item.get("author"),
                "published_at_utc": item.get("published_utc"),
                "source_domain": get_source_domain(item.get("article_url")),
                "source_url": item.get("article_url"),
                "canonical_url": get_canonical_url(item.get("article_url")),
                "tickers": item.get("tickers", []),
                "topics": topics,
                "language": "en",
                "content_hash": generate_content_hash(content)
            }))
            new_items += 1

        print(f"✅ Страница {page + 1}: добавлено {new_items} новостей")

        if new_items < page_limit:
            print("Похоже, достигнут конец списка новостей.")
            break

        offset += page_limit

    print(f"Polygon.io: всего найдено {len(news_list)} новостей")
    return news_list

# --- Alpha Vantage ---
def get_alpha_vantage_news_all(start_date, end_date, topics="finance,business,technology", limit=100, fetch_full_text=True):
    news_list = []
    start_norm = normalize_date_str(start_date)
    end_norm = normalize_date_str(end_date)
    time_from = datetime.strptime(start_norm, "%Y-%m-%d").strftime("%Y%m%dT0000")
    time_to = datetime.strptime(end_norm, "%Y-%m-%d").strftime("%Y%m%dT2359")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics={topics}&time_from={time_from}&time_to={time_to}&limit={limit}&sort=LATEST&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])
        for item in feed:
            print("Обработка новости:", item.get("title", "")[:60], " от Alpha Vantage")
            title = item.get("title", "")
            summary = item.get("summary", "")
            content = item.get("content", "")
            source_url = item.get("url", "")
            if fetch_full_text and source_url and (not content or "[+" in content or len(content)<200):
                full_text = parser_html(source_url)
                if full_text:
                    content = full_text
            time_str = item.get("time_published")
            pub = None
            for fmt in ("%Y%m%dT%H%M%S", "%Y%m%dT%H%M"):
                try:
                    pub = datetime.strptime(time_str, fmt)
                    break
                except:
                    continue
            if not pub:
                continue
            if content == "":
                continue
            topics_raw = item.get("topics", [])
            topics = [
                t["topic"] if isinstance(t, dict) and "topic" in t else t
                for t in topics_raw
            ]
            news_list.append(to_final_format({
                "api_source": "Alpha Vantage",
                "title": title,
                "summary": summary,
                "content": content,
                "author": ", ".join(item.get("authors", [])),
                "published_at_utc": pub.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_domain": get_source_domain(source_url),
                "source_url": source_url,
                "canonical_url": get_canonical_url(source_url),
                "tickers": [t.get("ticker") for t in item.get("ticker_sentiment", []) if t.get("ticker")] if item.get("ticker_sentiment") else [],
                "topics": topics,
                "language": "en",
                "content_hash": generate_content_hash(content)
            }))
        print(f"Alpha Vantage: найдено {len(news_list)} новостей")
    except Exception as e:
        print(f"Ошибка Alpha Vantage: {e}")
    return news_list

# --- FMP ---
def get_fmp_news_all(start_date, end_date, fetch_full_text=True):
    news_list = []
    seen_titles = set()  # Чтобы отслеживать дубликаты
    start_norm = normalize_date_str(start_date)
    end_norm = normalize_date_str(end_date)
    start_dt = datetime.strptime(start_norm, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_norm, "%Y-%m-%d").date()
    page = 0
    limit = 100
    max_pages = 50  # максимум 5000 новостей

    while page < max_pages:
        print(f"Запрос страницы {page + 1} новостей FMP...")
        url = f"https://financialmodelingprep.com/stable/fmp-articles?page={page}&limit={limit}&apikey={FMP_API_KEY}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"⚠️ Ошибка при запросе FMP (страница {page}): {e}")
            break

        if not data or not isinstance(data, list):
            print("FMP: пустой ответ или неверный формат данных")
            break

        count_added = 0
        for item in data:
            print("Обработка новости:", item.get("title", "")[:60], " от FMP")
            pub_dt = parse_date(item.get("publishedDate") or item.get("date"))
            if not pub_dt or not (start_dt <= pub_dt.date() <= end_dt):
                continue

            title = item.get("title", "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)

            content = strip_html(item.get("text") or item.get("content") or "")
            if fetch_full_text and item.get("link") and len(content) < 200:
                full_text = parser_html(item.get("link"))
                if full_text:
                    content = full_text
            if content == "":
                continue
            topics = []
            topics += extract_entities_universal(content)
            topics = remove_nested_duplicates(topics)
            topics = [topic for topic in topics if len(topic) <= MAX_TOPIC_LENGTH and not has_duplicate_words(topic)]
            news_list.append(to_final_format({
                "api_source": "Financial Modeling Prep",
                "title": title,
                "summary": None,
                "content": content,
                "author": item.get("author", ""),
                "published_at_utc": pub_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_domain": get_source_domain(item.get("link") or item.get("url")),
                "source_url": item.get("link") or item.get("url"),
                "canonical_url": get_canonical_url(item.get("link") or item.get("url")),
                "tickers": [item.get("tickers")] if item.get("tickers") else [],
                "topics": topics,
                "language": "en",
                "content_hash": generate_content_hash(content)
            }))
            count_added += 1

        print(f"✅ FMP страница {page + 1}: добавлено {count_added} новостей (всего {len(news_list)})")

        # Если добавлено меньше, чем лимит — скорее всего достигнут конец
        if count_added < limit:
            print("Похоже, достигнут конец ленты FMP.")
            break

        page += 1

    print(f"FMP: всего найдено {len(news_list)} новостей")
    return news_list

# --- NewsAPI ---
def get_newsapi_news(start_date, end_date, fetch_full_text=True, lan = 'en'):
    """Получает новости от NewsAPI со всеми доступными полями."""
    print("Запрос новостей от NewsAPI...")
    news_list = []

    try:
        start_norm = normalize_date_str(start_date)
        end_norm = normalize_date_str(end_date)
        start_dt = datetime.strptime(start_norm, "%Y-%m-%d")
        end_dt = datetime.strptime(end_norm, "%Y-%m-%d")
    except ValueError as e:
        print(f"Ошибка формата даты: {e}")
        return []

    today = datetime.now()
    one_month_ago = today - timedelta(days=30)

    if start_dt < one_month_ago:
        original_start = start_norm
        start_norm = one_month_ago.strftime("%Y-%m-%d")
        print(f"⚠️  NewsAPI бесплатный план: start_date скорректирован с {original_start} на {start_norm}")

    if end_dt > today:
        end_norm = today.strftime("%Y-%m-%d")

    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'finance OR business OR stock OR earnings OR economy',
        'from': start_norm,
        'to': end_norm,
        'sortBy': 'publishedAt',
        'language': lan,
        'pageSize': 100,
        'apiKey': NEWS_API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 426:
            print("NewsAPI 426: бесплатный план не поддерживает исторические данные старше 1 месяца.")
            return []

        response.raise_for_status()
        data = response.json()

        for item in data.get("articles", []):
            title = (item.get("title") or "").strip()
            source_url = (item.get("url") or "").strip()
            if not title or not source_url:
                continue
            print("Обработка новости:", item.get("title", "")[:60], " от NewsAPI")
            content = ""
            # --- Проверяем, есть ли "[+... chars]" ---
            if fetch_full_text:
                print(f"🔄 Загрузка полного текста статьи: {title[:60]}...")
                full_text = parser_html(source_url)
                if full_text:
                    content = full_text

            # безопасная проверка пустоты
            if not content.strip():
                continue
            topics = []
            topics += extract_entities_universal(content)
            topics = remove_nested_duplicates(topics)
            topics = [topic for topic in topics if len(topic) <= MAX_TOPIC_LENGTH and not has_duplicate_words(topic)]
            news_list.append(to_final_format({
                    "api_source": "NewsAPI",
                    "title": item.get("title", ""),
                    "summary": item.get("description", ""),
                    "content": content.strip(),
                    "author": item.get("author"),
                    "published_at_utc": item.get("publishedAt"),
                    "source_domain": get_source_domain(item.get("url")),
                    "source_url": item.get("url"),
                    "canonical_url": get_canonical_url(item.get("url")),
                    "tickers": [],
                    "topics": topics,
                    "language": lan,
                    "content_hash": generate_content_hash(content)
                }))
        print(f"NewsAPI: получено {len(news_list)} новостей")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе к NewsAPI: {e}")

    return news_list

def strip_html(html_str):
    if not html_str:
        return ""
    clean = re.sub(r'<[^>]+>', '', html_str)
    clean = clean.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return re.sub(r'\s+', ' ', clean).strip()


# from datetime import datetime, timedelta
#
#
# def fetch_news_by_day_range(api_func, start_date_str, end_date_str, **kwargs):
#     """Fetches news day by day, working backwards from end_date to start_date."""
#     all_results = []
#
#     try:
#         end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
#         start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
#     except ValueError as e:
#         print(f"❌ Ошибка формата даты: {e}")
#         return []
#
#     current_end = end_date
#
#     while current_end >= start_date:
#         current_start = current_end - timedelta(days=1)
#
#         # Не уходим за пределы start_date
#         if current_start < start_date:
#             current_start = start_date
#
#         start_str = current_start.strftime("%Y-%m-%d")
#         end_str = current_end.strftime("%Y-%m-%d")
#
#         print(f"\n📅 Запрос диапазона: {start_str} → {end_str}")
#
#         try:
#             daily_news = api_func(start_str, end_str, **kwargs)
#
#             if not daily_news:
#                 print(f"⚠️ Пустой ответ для {start_str} → {end_str}, прерываем")
#                 break
#
#             all_results.extend(daily_news)
#             print(f"✅ Получено {len(daily_news)} новостей за этот день")
#
#         except Exception as e:
#             print(f"❌ Ошибка при запросе {start_str} → {end_str}: {e}")
#             break
#
#         # Сдвигаемся на день назад
#         current_end = current_start - timedelta(days=1)
#
#         if current_end < start_date:
#             break
#
#     return all_results
#
#
# if __name__ == "__main__":
#     all_news = []
#
#     if POLYGON_API_KEY:
#         all_news.extend(fetch_news_by_day_range(
#             get_polygon_news_all, start_date_str, end_date_str
#         ))
#
#     if ALPHA_VANTAGE_API_KEY:
#         all_news.extend(fetch_news_by_day_range(
#             get_alpha_vantage_news_all, start_date_str, end_date_str
#         ))
#
#     if FMP_API_KEY:
#         all_news.extend(fetch_news_by_day_range(
#             get_fmp_news_all, start_date_str, end_date_str
#         ))
#
#     if NEWS_API_KEY:
#         all_news.extend(fetch_news_by_day_range(
#             get_newsapi_news, start_date_str, end_date_str
#         ))
#         all_news.extend(fetch_news_by_day_range(
#             get_newsapi_news, start_date_str, end_date_str, lan='ru'
#         ))
#
#     all_news.sort(key=lambda x: x['published_time'], reverse=True)
#     print(f"\n✅ Всего новостей: {len(all_news)}")
#
#     if all_news:
#         print(json.dumps(all_news[0], indent=4, ensure_ascii=False))
#
#     with open("all_news.json", "w", encoding="utf-8") as f:
#         json.dump(all_news, f, indent=4, ensure_ascii=False)

# --- Главный сбор ---
if __name__ == "__main__":
    all_news = []
    if POLYGON_API_KEY:
        all_news.extend(get_polygon_news_all(start_date_str, end_date_str))
    if ALPHA_VANTAGE_API_KEY:
        all_news.extend(get_alpha_vantage_news_all(start_date_str, end_date_str))
    if FMP_API_KEY:
        all_news.extend(get_fmp_news_all(start_date_str, end_date_str))
    if NEWS_API_KEY:
        all_news.extend(get_newsapi_news(start_date_str, end_date_str))
        all_news.extend(get_newsapi_news(start_date_str, end_date_str, lan='ru'))

    all_news.sort(key=lambda x: x['published_time'], reverse=True)
    print(f"\n✅ Всего новостей: {len(all_news)}")
    if all_news:
        print(json.dumps(all_news[0], indent=4, ensure_ascii=False))
    with open("all_api_news_last_24.json", "w", encoding="utf-8") as f:
        json.dump(all_news, f, indent=4, ensure_ascii=False)