import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from razdel import tokenize as razdel_tokenize
import nltk
import ssl

# Настройка SSL для загрузки данных NLTK
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Загрузка необходимых данных для NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Инициализация стоп-слов и лемматизатора
stop_words = set(stopwords.words('russian'))
lemmatizer = WordNetLemmatizer()

# Функция для очистки и предобработки текста
def preprocess_text(text):
    # Проверка на NaN и замена на пустую строку, если это NaN
    if pd.isna(text):
        return ""
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление спецсимволов
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s]', '', text)
    # Токенизация с использованием razdel
    words = [token.text for token in razdel_tokenize(text)]
    # Удаление стоп-слов и лемматизация
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Загрузка данных
reviews_df = pd.read_csv("hotel_reviews.csv")

# Применение функции предобработки к тексту положительных и отрицательных отзывов
reviews_df['cleaned_positive_text'] = reviews_df['Положительный отзыв'].apply(preprocess_text)
reviews_df['cleaned_negative_text'] = reviews_df['Отрицательный отзыв'].apply(preprocess_text)

# Объединение очищенных положительных и отрицательных отзывов в один столбец
reviews_df['cleaned_text'] = reviews_df['cleaned_positive_text'] + ' ' + reviews_df['cleaned_negative_text']

# Округление оценок и преобразование к диапазону 0–9
reviews_df['adjusted_labels'] = reviews_df['Оценка'].round().astype(int) - 1

# Сохранение предобработанного датасета с корректными метками для классификации
reviews_df[['adjusted_labels', 'cleaned_text']].to_csv("cleaned_booking_reviews.csv", index=False)
print("Предобработка завершена. Данные сохранены в cleaned_booking_reviews.csv")
