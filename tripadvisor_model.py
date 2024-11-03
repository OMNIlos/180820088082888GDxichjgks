import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss

# Шаг 1: Загрузка и предобработка данных
print("Загрузка данных...")
df = pd.read_csv("tripadvisor_hotel_reviews.csv")
df.columns = ['review', 'rating']
df['adjusted_rating'] = (df['rating'] - 1) * 2
print("Данные загружены.")

# Разделение данных на обучающую и тестовую выборки
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'].tolist(),
    df['adjusted_rating'].tolist(),
    test_size=0.2,
    random_state=42
)
print("Данные разделены на обучающую и тестовую выборки.")

# Шаг 2: Инициализация токенизатора и модели
print("Загрузка токенизатора и модели...")
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=10)
print("Токенизатор и модель загружены.")

# Шаг 3: Настройка весов классов для функции потерь
class_counts = df['adjusted_rating'].value_counts().reindex(range(10), fill_value=0)
class_weights = 1.0 / (class_counts + 1)  # Избегаем деления на 0, добавив 1
class_weights = class_weights / class_weights.sum()  # Нормализация весов
class_weights = torch.tensor(class_weights.values, dtype=torch.float).to(model.device)

# Использование CrossEntropyLoss с весами
loss_fn = CrossEntropyLoss(weight=class_weights)
print("Функция потерь настроена с учетом весов классов.")

# Шаг 4: Создание класса Dataset для загрузки данных
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

# Шаг 5: Создание DataLoader с num_workers=0
print("Создание DataLoader для обучающей и тестовой выборок...")
train_dataset = ReviewsDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewsDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)
print("DataLoader успешно создан.")

# Настройка оптимизатора и устройства
optimizer = AdamW(model.parameters(), lr=2e-5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
print("Модель настроена для обучения на устройстве:", device)

# Шаг 6: Обучение модели
num_epochs = 3
model.train()

for epoch in range(num_epochs):
    print(f"Начало эпохи {epoch + 1}")
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        total_loss += loss.item()

        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()

        # Вычисление точности на обучающей выборке
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

        # Периодический вывод для контроля
        if i % 50 == 0:
            print(f"Batch {i}: текущая потеря {loss.item()}")

    # Среднее значение потерь и точность для текущей эпохи
    avg_loss = total_loss / len(train_loader)
    train_accuracy = correct_predictions.double() / total_predictions
    print(f"Эпоха {epoch + 1}/{num_epochs}")
    print(f"Средняя потеря: {avg_loss:.4f}")
    print(f"Точность на обучающей выборке: {train_accuracy:.4f}")
    print("-" * 30)

# Шаг 7: Оценка модели на тестовых данных
model.eval()
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        total_predictions += labels.size(0)

test_accuracy = correct_predictions.double() / total_predictions
print(f"\nТочность на тестовой выборке: {test_accuracy:.4f}")

# Шаг 8: Сохранение модели, токенизатора и оптимизатора
save_path = " ... " #сюда необходимо вставить путь до папки с весами модели - tripadvisor_hotel_model.
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.pt'))

print(f"\nМодель и токенизатор успешно сохранены в {save_path}")
