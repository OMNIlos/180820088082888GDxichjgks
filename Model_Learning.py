import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import signal

# Указание пути для сохранения модели
save_path = " ... " #сюда необходимо вставить путь до папки с весами модели - upbringing_babymodel,
os.makedirs(save_path, exist_ok=True)

# Функция тайм-аута для обучения
def handler(signum, frame):
    raise TimeoutError("Время выполнения обучения истекло")

# Устанавливаем тайм-аут на 60 секунд
signal.signal(signal.SIGALRM, handler)
signal.alarm(60)

try:
    # Шаг 1: Загрузка и подготовка данных с ограничением на размер для быстрого тестирования
    reviews_df = pd.read_csv("cleaned_booking_reviews.csv")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        reviews_df['cleaned_text'].tolist()[:100],  # Ограничение на 100 примеров для проверки
        reviews_df['adjusted_labels'].tolist()[:100],
        test_size=0.2,
        random_state=42
    )

    # Загрузка токенизатора и модели
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=10)

    # Класс Dataset для загрузки данных
    class ReviewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    # Создание DataLoader для загрузки данных
    train_dataset = ReviewsDataset(train_texts, train_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Настройка оптимизатора
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Обучение модели
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.train()

    num_epochs = 3

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Обратное распространение ошибки и обновление весов
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Средняя потеря для эпохи {epoch + 1}: {avg_loss}")

    # Сохранение модели, токенизатора и оптимизатора с отладочными сообщениями
    try:
        model.save_pretrained(save_path)
        print("Модель успешно сохранена в:", save_path)
    except Exception as e:
        print("Ошибка при сохранении модели:", e)

    try:
        tokenizer.save_pretrained(save_path)
        print("Токенизатор успешно сохранен в:", save_path)
    except Exception as e:
        print("Ошибка при сохранении токенизатора:", e)

    try:
        torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))
        print("Состояние оптимизатора успешно сохранено в:", save_path)
    except Exception as e:
        print("Ошибка при сохранении состояния оптимизатора:", e)

    # Сброс тайм-аута, если выполнение прошло успешно
    signal.alarm(0)

except TimeoutError as e:
    print(e)
