import { useState, useEffect } from 'react';
import { fetchNews } from '../../api';
import './MainPage.css';

function MainPage() {
  const [startDate, setStartDate] = useState('2024-10-10');
  const [endDate, setEndDate] = useState('2025-10-04');
  const [news, setNews] = useState([]);
  const [filteredNews, setFilteredNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedCard, setExpandedCard] = useState(null);

  const loadNews = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await fetchNews(startDate, endDate);
      console.log('Загруженные данные:', data);
      setNews(Array.isArray(data) ? data : []);
      filterNewsByDate(Array.isArray(data) ? data : [], startDate, endDate);
    } catch (err) {
      console.error('Ошибка:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const filterNewsByDate = (newsData, start, end) => {
    const startTimestamp = new Date(start).getTime();
    const endTimestamp = new Date(end).getTime();

    const filtered = newsData.filter(item => {
      const newsDate = item.sources?.[0]?.time || item.timeline?.[0]?.time;
      if (!newsDate) return true; // Показываем новости без даты

      const newsTimestamp = new Date(newsDate).getTime();
      return newsTimestamp >= startTimestamp && newsTimestamp <= endTimestamp;
    });

    setFilteredNews(filtered);
  };

  useEffect(() => {
    loadNews();
  }, []);

  const handleApplyFilter = () => {
    filterNewsByDate(news, startDate, endDate);
  };

  const getHotnessClass = (hotness) => {
    if (hotness >= 0.7) return 'hotness-high';
    if (hotness >= 0.4) return 'hotness-medium';
    return 'hotness-low';
  };

  const toggleOverlay = (index) => {
    setExpandedCard(expandedCard === index ? null : index);
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Дата неизвестна';
    const date = new Date(dateString);
    return date.toLocaleDateString('ru-RU', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="page-container">
      <div className="date-filter">
        <h2>Фильтр по датам</h2>
        <div className="date-inputs">
          <div className="date-field">
            <label htmlFor="start-date">От:</label>
            <input
              type="date"
              id="start-date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </div>
          <div className="date-field">
            <label htmlFor="end-date">До:</label>
            <input
              type="date"
              id="end-date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </div>
          <button onClick={handleApplyFilter} disabled={loading}>
            {loading ? 'Загрузка...' : 'Применить'}
          </button>
        </div>
      </div>

      {error && <div className="error">Ошибка: {error}</div>}

      <div className="main-page">
        {loading ? (
          <div className="loading">Загрузка новостей...</div>
        ) : filteredNews.length === 0 ? (
          <div className="no-data">Нет новостей для отображения в выбранном диапазоне</div>
        ) : (
          filteredNews.map((item, index) => (
            <div key={index} className="news-card">
              <div className={`news-hotness ${getHotnessClass(item.hotness)}`}>
                Hotness: {(item.hotness * 100).toFixed(0)}%
              </div>

              <div className="news-date">
                {formatDate(item.sources?.[0]?.time || item.timeline?.[0]?.time)}
              </div>

              <h3 className="news-headline">{item.headline}</h3>
              <p className="news-why">{item.why_now}</p>

              <div className="entities">
                {item.entities?.map((entity, i) => (
                  <span key={i} className="entity-tag">{entity}</span>
                ))}
              </div>

              <div className="source-line">
                Источник:
                <a href={item.sources?.[0]?.url} target="_blank" rel="noopener noreferrer">
                  {item.sources?.[0]?.type || 'ссылка'}
                </a>
              </div>

              <div className="card-footer">
                <button
                  className="details-btn"
                  onClick={() => toggleOverlay(index)}
                >
                  Показать детали
                </button>
              </div>

              {expandedCard === index && (
                <div className="overlay" onClick={() => toggleOverlay(index)}>
                  <div className="overlay-content" onClick={(e) => e.stopPropagation()}>
                    <h4 className="draft-title">{item.draft?.title}</h4>
                    <p className="draft-lead">{item.draft?.lead}</p>
                    {item.draft?.bullets && (
                      <ul>
                        {item.draft.bullets.map((bullet, i) => (
                          <li key={i}>{bullet}</li>
                        ))}
                      </ul>
                    )}
                    <button className="details-btn red" onClick={() => toggleOverlay(index)}>
                      Закрыть
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default MainPage;
