from datetime import datetime, timedelta, timezone
from  link_parser import get_links, search, build_query
from parser import get_data
import json
from dateutil import parser

queries = [
    'финансы OR бизнес "новости"',
    '"бизнес новости" OR "экономика"',
    '"акции" OR "фондовый рынок"',
    '"инвестиции" OR "инвестирование"',
    '"корпоративные новости" OR "стартапы" ',
    '"рынок" OR "финансовый рынок" "новости"',
    '"макроэкономика" OR "экономический прогноз" OR "финансы" ',
    '"банковский сектор" OR "криптовалюты" OR "фондовый рынок"',
    '"рост" OR "падение" OR "увеличение" -ЧС'
           ]
flags = " '-видео' '-все новости по теме' '-Инвестиции — последние новости сегодня на РБК.Ру' '-свежие новости рынков и инвестиций' '-последние новости'"

def to_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main(time_end_str):
    time_end = datetime.fromisoformat(time_end_str).replace(tzinfo=timezone.utc)
    time_start = time_end - timedelta(days=1)

    links1 = get_links(count=100)
    links2 = []
    res = []

    for q in queries:
        q_full = build_query(q + flags, (time_start.strftime("%Y-%m-%d"), time_end.strftime("%Y-%m-%d")))
        links2 += search(q_full, 10)
    print("Количество запросов:", len(queries))
    print("Ссылок из RSS:", len(links1))
    print("Ссылок из поиска:", len(links2))

    links = links1 + links2
    print("Всего ссылок:", len(links))

    for link in links:
        print(link)
        data = get_data(link)
        if data:
            res.append(data)

    res_filtered = []

    for js in res:
        published_str = js.get("published_time")
        if not published_str:
            continue

        try:
            time_news = parser.isoparse(published_str)
            time_news = to_utc(time_news)
        except Exception as e:
            print(f"[WARN] Ошибка парсинга даты '{published_str}': {e}")
            continue

        if time_start <= time_news <= time_end:
            res_filtered.append(js)

    res = res_filtered
    print("Получено ссылок:", len(links))
    print(f"Всего за сутки: {len(res)}")

    with open("apigoogle_last24.jsongoogle_last24.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


main("2025-10-04T23:59:00") # Время конца сегодняшнего дня
