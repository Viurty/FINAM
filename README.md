# FastAPI News Aggregator

Веб-приложение для агрегации и отображения новостей из различных источников с использованием FastAPI и React.

## Структура проекта

```
fastapi-app/
├── backend/
│   ├── google_api.py       # Парсер новостей через Google API
│   ├── all_api_news.py     # Парсер новостей из других источников
│   ├── join.py             # Объединение данных в единый файл
│   └── requirements.txt
└── frontend/
    └── public/
        └── news-index.json # Итоговый файл с новостями
```

## Требования

- Python 3.9+
- Node.js 14+ / npm или yarn
- pip

## Установка

### Backend

1. Создайте виртуальное окружение:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# или .venv\Scripts\activate  # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
# или
yarn install
```

## Запуск

### Backend (сбор новостей)

Запускайте скрипты в следующем порядке:

1. Сбор новостей из Google API:
```bash
cd backend
source .venv/bin/activate
python google_api.py
```

2. Сбор новостей из других источников:
```bash
python all_api_news.py
```

3. Объединение данных:
```bash
python join.py
```

Результат будет сохранён в `frontend/public/news-index.json`.

### Frontend

```bash
cd frontend
npm start
# или
yarn start
```

Приложение будет доступно по адресу `http://localhost:3000`.


## requirements.txt

```txt
fastapi==0.104.1
uvicorn==0.24.0
requests==2.31.0
beautifulsoup4==4.12.2
tldextract==5.1.0
torch==2.1.0
transformers==4.35.0
newspaper3k==0.2.8
python-dotenv==1.0.0
aiohttp==3.9.0
pydantic==2.5.0
lxml==4.9.3
```

## Особенности

- **Фильтрация по датам**: доступна на фронтенде для просмотра новостей за определенный период
- **Детальный просмотр**: каждая новость может быть раскрыта для просмотра полной информации
- **Автоматическое обновление**: запускайте backend-скрипты периодически (например, через cron) для обновления новостей

## Устранение проблем

### ModuleNotFoundError
Убедитесь, что виртуальное окружение активировано и все зависимости установлены:
```bash
pip install -r requirements.txt
```

### PyTorch/TensorFlow не найдены
Для работы NER-моделей установите PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```