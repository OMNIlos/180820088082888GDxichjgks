import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# Ввод количества батчей и эпох
input_batches = int(input("Ввидите количество батчей: "))
input_epochs = int(input("Ввидите количество эпох: "))

# Указание пути к сохраненной модели и токенизатору
save_path = " ... " #сюда необходимо вставить путь до папки с весами модели - upbringing_babymodel,

# Загрузка данных для оценки или дообучения
reviews_df = pd.read_csv("cleaned_booking_reviews.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    reviews_df['cleaned_text'].tolist(),
    reviews_df['adjusted_labels'].tolist(),
    test_size=0.2,
    random_state=42
)

# Загрузка токенизатора и модели из сохраненной директории
tokenizer = BertTokenizer.from_pretrained(save_path)
model = BertForSequenceClassification.from_pretrained(save_path)

# Класс Dataset для подготовки данных
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
val_dataset = ReviewsDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Настройка оптимизатора
optimizer = AdamW(model.parameters(), lr=2e-5)

# Попытка загрузить сохраненное состояние оптимизатора
try:
    optimizer.load_state_dict(torch.load(os.path.join(save_path, 'optimizer.pt'), weights_only=True))
    print("Состояние оптимизатора загружено успешно.")
except Exception as e:
    print("Ошибка при загрузке состояния оптимизатора:", e)

# Обучения или оценка
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# Ограничение на количество батчей для теста
max_batches = input_batches

# Продолжение обучения
num_epochs = input_epochs
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for i, batch in enumerate(train_loader):
        if i >= max_batches:  # Прерывание после max_batches
            break

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

    avg_loss = total_loss / max_batches
    print(f"Средняя потеря для эпохи {epoch + 1}: {avg_loss}")

# Оценка модели на тестовых данных
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        if i >= max_batches:  # Прерывание после max_batches
            break

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

accuracy = correct_predictions.double() / total_predictions
print(f"Точность на тестовых данных: {accuracy:.4f}")
