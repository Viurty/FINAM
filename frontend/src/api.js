const API_BASE_URL = 'http://localhost:8000';
const USE_TEST_DATA = true;

export const fetchNews = async (startDate, endDate) => {
  try {
    if (USE_TEST_DATA) {
      // Загружаем индексный файл со списком JSON
      const indexResponse = await fetch('/news-index.json');

      if (!indexResponse.ok) {
        throw new Error(`Failed to load index: ${indexResponse.status}`);
      }

      const index = await indexResponse.json();
      const allNews = [];

      // Загружаем все файлы параллельно
      const newsPromises = index.files.map(async (filename) => {
        try {
          const response = await fetch(`/${filename}`);
          if (!response.ok) {
            console.warn(`Failed to load ${filename}`);
            return [];
          }
          const data = await response.json();

          // Нормализуем данные
          if (Array.isArray(data)) {
            return data;
          } else if (data.news && Array.isArray(data.news)) {
            return data.news;
          } else {
            return [data];
          }
        } catch (error) {
          console.error(`Error loading ${filename}:`, error);
          return [];
        }
      });

      const newsArrays = await Promise.all(newsPromises);

      // Объединяем все массивы новостей
      newsArrays.forEach(newsArray => {
        allNews.push(...newsArray);
      });

      // Сортируем по дате (новые сверху)
      allNews.sort((a, b) => {
        const dateA = new Date(a.sources?.[0]?.time || a.timeline?.[0]?.time || 0);
        const dateB = new Date(b.sources?.[0]?.time || b.timeline?.[0]?.time || 0);
        return dateB - dateA;
      });

      return allNews;
    }

    const response = await fetch(
      `${API_BASE_URL}/news?start_date=${startDate}&end_date=${endDate}`
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Ошибка загрузки новостей:', error);
    throw error;
  }
};
