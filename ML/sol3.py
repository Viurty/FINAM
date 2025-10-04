import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from datetime import datetime
import json
from typing import List, Dict
import numpy as np


class FinancialNewsRadar(nn.Module):
    """
    Компактная модель для выявления горячих финансовых новостей.
    Использует предобученный трансформер + головы классификации.
    """

    def __init__(self, model_name='DeepPavlov/rubert-base-cased-sentence', hidden_dim=768):
        super().__init__()

        # Базовая модель для embeddings
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Голова для hotness score (0-1)
        self.hotness_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # Голова для entity extraction (multi-label)
        self.entity_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)  # динамический размер под найденные entities
        )

        # Голова для срочности (why_now)
        self.urgency_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # 5 категорий срочности
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        # Игнорируем лишние аргументы через **kwargs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        hotness = self.hotness_head(pooled)
        entity_logits = self.entity_head(pooled)
        urgency = self.urgency_classifier(pooled)

        return {
            'hotness': hotness,
            'entity_logits': entity_logits,
            'urgency': urgency,
            'embeddings': pooled
        }


class NewsProcessor:
    """Обработчик новостей с дедупликацией и ранжированием"""

    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)

        # Справочники для entities (можно расширить)
        self.known_entities = {
            'banks': ['Сбербанк', 'ВТБ', 'Альфа', 'Тинькофф'],
            'sectors': ['Banking', 'Energy', 'Tech', 'Retail'],
            'regulators': ['ЦБ', 'Минфин', 'ФНС']
        }

    def extract_features(self, news_item: Dict) -> Dict:
        """Извлечение признаков из новости"""
        text = f"{news_item.get('title', '')} {news_item.get('text', '')}"

        # Токенизация БЕЗ token_type_ids
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Удаляем token_type_ids если они есть
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Временные признаки
        timestamp = datetime.fromisoformat(news_item.get('timestamp', datetime.now().isoformat()))
        time_features = {
            'recency': (datetime.now() - timestamp).seconds / 3600,
            'hour_of_day': timestamp.hour,
            'is_market_hours': 10 <= timestamp.hour <= 19
        }

        source_weight = self._get_source_weight(news_item.get('source', ''))

        return {
            'inputs': inputs,
            'time_features': time_features,
            'source_weight': source_weight,
            'raw_text': text
        }

    def _get_source_weight(self, source: str) -> float:
        """Вес источника (официальные релизы важнее слухов)"""
        weights = {
            'release': 1.0,
            'bloomberg': 0.9,
            'reuters': 0.9,
            'interfax': 0.8,
            'telegram': 0.5,
            'unknown': 0.3
        }
        for key, weight in weights.items():
            if key in source.lower():
                return weight
        return 0.3

    def calculate_hotness(self, news_item: Dict, model_output: Dict, cluster_info: Dict) -> float:
        """Комплексный расчет hotness с учетом всех факторов"""
        features = self.extract_features(news_item)

        # Базовый score от модели
        base_hotness = model_output['hotness'].item()

        # Бонусы за подтверждения
        confirmation_bonus = min(cluster_info.get('num_sources', 1) * 0.1, 0.3)

        # Штраф за время
        recency_penalty = max(0, 1 - features['time_features']['recency'] / 24)

        # Бонус за рыночные часы
        market_bonus = 0.1 if features['time_features']['is_market_hours'] else 0

        # Вес источника
        source_multiplier = features['source_weight']

        final_hotness = (
                base_hotness * source_multiplier * recency_penalty +
                confirmation_bonus + market_bonus
        )

        return min(final_hotness, 1.0)

    def extract_entities(self, text: str) -> List[str]:
        """Простое извлечение entities через совпадения"""
        entities = []
        text_lower = text.lower()

        for category, items in self.known_entities.items():
            for item in items:
                if item.lower() in text_lower:
                    entities.append(item)

        return list(set(entities))

    def generate_why_now(self, cluster_info: Dict, urgency_class: int) -> str:
        """Генерация объяснения актуальности"""
        templates = {
            0: "Рутинная информация без особой срочности",
            1: f"Появилось {cluster_info.get('num_sources', 1)} подтверждающих источника за последний час",
            2: f"Официальный релиз + {cluster_info.get('num_sources', 1)} медиа-подтверждений за {cluster_info.get('time_span', 2)} часа",
            3: "Неожиданное событие с потенциальным влиянием на ликвидность/волатильность",
            4: "КРИТИЧНО: Резкое изменение рыночной ситуации, требует немедленного внимания"
        }
        return templates.get(urgency_class, templates[1])

    def deduplicate_and_cluster(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """Дедупликация и кластеризация похожих новостей"""
        clusters = {}
        embeddings_cache = {}

        for news in news_list:
            features = self.extract_features(news)
            with torch.no_grad():
                output = self.model(**features['inputs'])
                emb = output['embeddings'].cpu().numpy()

            # Поиск похожего кластера (cosine similarity > 0.85)
            assigned = False
            for cluster_id, cluster_data in clusters.items():
                centroid = cluster_data['centroid']
                similarity = np.dot(emb[0], centroid) / (
                        np.linalg.norm(emb[0]) * np.linalg.norm(centroid)
                )

                if similarity > 0.85:
                    cluster_data['news'].append(news)
                    cluster_data['centroid'] = (
                                                       cluster_data['centroid'] * len(cluster_data['news']) + emb[0]
                                               ) / (len(cluster_data['news']) + 1)
                    assigned = True
                    break

            if not assigned:
                cluster_id = f"cluster_{len(clusters)}"
                clusters[cluster_id] = {
                    'news': [news],
                    'centroid': emb[0]
                }

        return clusters


class DraftGenerator:
    """Генератор черновиков публикаций"""

    def generate_draft(self, event_data: Dict) -> Dict:
        """Создание черновика поста"""
        headline = event_data['headline']
        entities = event_data['entities']
        sources = event_data['sources']

        # Формирование заголовка
        title = headline

        # Lead paragraph
        lead = self._generate_lead(event_data)

        # Bullets
        bullets = [
            f"{len(sources)} подтверждающих источника",
            f"Затронутые активы: {', '.join(entities[:3])}",
            f"Hotness score: {event_data['hotness']:.2f}"
        ]

        note = "Детальный таймлайн и ссылки приведены выше."

        return {
            'title': title,
            'lead': lead,
            'bullets': bullets,
            'note': note
        }

    def _generate_lead(self, event_data: Dict) -> str:
        """Генерация вводного параграфа"""
        entities_str = ', '.join(event_data['entities'][:3])
        return (
            f"Сегодня появилась информация о {event_data['headline'].lower()}. "
            f"Событие затрагивает {entities_str} и может повлиять на рыночную динамику. "
            f"Мы отслеживаем развитие ситуации."
        )


def process_news_batch(news_data: List[Dict], model: FinancialNewsRadar,
                       processor: NewsProcessor, top_k: int = 5) -> List[Dict]:
    """Основной пайплайн обработки новостей"""

    # 1. Дедупликация и кластеризация
    clusters = processor.deduplicate_and_cluster(news_data)

    results = []
    draft_gen = DraftGenerator()

    # 2. Обработка каждого кластера
    for cluster_id, cluster_data in clusters.items():
        # Берем самую свежую/авторитетную новость как основную
        main_news = sorted(
            cluster_data['news'],
            key=lambda x: (
                processor._get_source_weight(x.get('source', '')),
                x.get('timestamp', '')
            ),
            reverse=True
        )[0]

        # 3. Извлечение фичей и предсказание
        features = processor.extract_features(main_news)
        with torch.no_grad():
            output = model(**features['inputs'])

        # 4. Расчет hotness
        cluster_info = {
            'num_sources': len(cluster_data['news']),
            'time_span': 2  # упрощение
        }
        hotness = processor.calculate_hotness(main_news, output, cluster_info)

        # 5. Извлечение entities
        entities = processor.extract_entities(features['raw_text'])

        # 6. Why now
        urgency_class = torch.argmax(output['urgency']).item()
        why_now = processor.generate_why_now(cluster_info, urgency_class)

        # 7. Формирование результата
        event = {
            'headline': main_news.get('title', 'Без заголовка'),
            'hotness': round(hotness, 2),
            'why_now': why_now,
            'entities': entities,
            'sources': [
                {
                    'type': 'media',
                    'url': n.get('url', ''),
                    'time': n.get('timestamp', '')
                } for n in cluster_data['news']
            ],
            'timeline': [
                {
                    'event': 'first_mention' if i == 0 else 'confirmation',
                    'time': n.get('timestamp', ''),
                    'url': n.get('url', '')
                } for i, n in enumerate(sorted(
                    cluster_data['news'],
                    key=lambda x: x.get('timestamp', '')
                ))
            ],
            'dedup_group': cluster_id
        }

        # 8. Генерация черновика
        event['draft'] = draft_gen.generate_draft(event)

        results.append(event)

    # 9. Сортировка по hotness и возврат top-K
    results.sort(key=lambda x: x['hotness'], reverse=True)
    return results[:top_k]


import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class FinancialNewsDataset(Dataset):
    """Dataset для обучения"""

    def __init__(self, data_path, tokenizer):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = f"{item.get('title', '')} {item.get('text', '')}"

        inputs = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Лейблы (нужно подготовить заранее)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'hotness': torch.tensor(item.get('hotness_label', 0.5)),
            'urgency': torch.tensor(item.get('urgency_label', 1))
        }


def train_model(model, train_loader, epochs=3, lr=2e-5):
    """Быстрое обучение для хакатона"""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    hotness_criterion = nn.MSELoss()
    urgency_criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )

            # Комбинированная функция потерь
            loss_hotness = hotness_criterion(
                outputs['hotness'].squeeze(),
                batch['hotness']
            )
            loss_urgency = urgency_criterion(
                outputs['urgency'],
                batch['urgency']
            )

            loss = loss_hotness + 0.5 * loss_urgency
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    return model


# Инициализация
model = FinancialNewsRadar()
processor = NewsProcessor(model, model.tokenizer)

# Загрузка данных
with open('data.json', 'r', encoding='utf-8') as f:
    news_data = json.load(f)

# Обработка
top_events = process_news_batch(news_data, model, processor, top_k=5)

# Сохранение результатов
with open('radar_output.json', 'w', encoding='utf-8') as f:
    json.dump(top_events, f, ensure_ascii=False, indent=2)