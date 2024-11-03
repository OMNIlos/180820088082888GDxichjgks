import pandas as pd
from transformers import BertTokenizer

# Загрузка токенизатора BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
reviews_df = pd.read_csv("cleaned_booking_reviews.csv")
# Токенизация текстов
tokens = tokenizer(list(reviews_df['cleaned_text']), padding=True, truncation=True, return_tensors='pt')
print(tokens)
