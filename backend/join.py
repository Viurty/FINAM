import json
from datetime import datetime
import os

def merge_json_files(file1_path, file2_path):
    with open(file1_path, 'r', encoding='utf-8') as f1:
        data1 = json.load(f1)
    with open(file2_path, 'r', encoding='utf-8') as f2:
        data2 = json.load(f2)

    merged_dict = {item['url']: item for item in data1 + data2}
    merged_list = list(merged_dict.values())

    date_str = None
    for item in merged_list:
        if 'published_time' in item and item['published_time']:
            try:
                parsed_date = datetime.fromisoformat(item['published_time'])
                date_str = parsed_date.strftime('%Y-%m-%d')
                break
            except ValueError:
                date_str = item['published_time'][:10]
                break

    output_path = "../frontend/public/" + os.path.join(f"{date_str}.json")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(merged_list, f_out, ensure_ascii=False, indent=4)

    print(f"Объединено {len(merged_list)} записей. Результат сохранён в: {output_path}")


def main():
    merge_json_files("api_google_last24.json", "all_api_news_last_24.json")


main()