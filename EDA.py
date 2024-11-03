import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from nltk.probability import FreqDist
from nltk.util import ngrams
import re
import nltk

# Загрузка данных
reviews_df = pd.read_csv("cleaned_booking_reviews.csv")

# Переименование столбцов
reviews_df = reviews_df.rename(columns={'adjusted_labels': 'Оценка', 'cleaned_text': 'cleaned_text'})

# Обработка пропусков и очистка данных
if reviews_df['Оценка'].isna().sum() > 0 or reviews_df['cleaned_text'].isna().sum() > 0:
    print("Есть пропуски в столбцах 'Оценка' или 'cleaned_text'. Проверьте данные.")
    reviews_df = reviews_df.dropna(subset=['Оценка', 'cleaned_text'])

# Удаление дубликатов
reviews_df = reviews_df.drop_duplicates()

# Очистка текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s]', '', text)
    return text

reviews_df['cleaned_text'] = reviews_df['cleaned_text'].apply(clean_text)

# Анализ длины текстов
reviews_df['review_length'] = reviews_df['cleaned_text'].apply(len)
average_length = reviews_df['review_length'].mean()
min_length = reviews_df['review_length'].min()
max_length = reviews_df['review_length'].max()

print(f"Средняя длина отзыва: {average_length}")
print(f"Минимальная длина отзыва: {min_length}")
print(f"Максимальная длина отзыва: {max_length}")

# Визуализация распределения длины текстов
plt.figure(figsize=(8, 6))
plt.hist(reviews_df['review_length'], bins=30, edgecolor='black')
plt.xlabel('Длина отзыва')
plt.ylabel('Частота')
plt.title('Распределение длины отзывов')
plt.show()

# Частотный анализ слов
stop_words = set(stopwords.words("english"))
all_words = ' '.join(reviews_df['cleaned_text'].tolist())
words = [word for word in all_words.split() if word not in stop_words]
word_freq = FreqDist(words)

# Облако слов
wordcloud = WordCloud(width=800, height=400, max_words=100, background_color='white').generate(' '.join(words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Облако слов")
plt.show()

# Частотный анализ биграмм
vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=20)
X_bigrams = vectorizer.fit_transform(reviews_df['cleaned_text'])
bigram_freq = dict(zip(vectorizer.get_feature_names_out(), X_bigrams.toarray().sum(axis=0)))
bigram_df = pd.DataFrame(list(bigram_freq.items()), columns=['bigram', 'frequency']).sort_values(by='frequency', ascending=False)

# Визуализация частот биграмм
plt.figure(figsize=(10, 5))
plt.barh(bigram_df['bigram'], bigram_df['frequency'])
plt.title('Частотный анализ биграмм')
plt.xlabel('Частота')
plt.ylabel('Биграмма')
plt.show()
