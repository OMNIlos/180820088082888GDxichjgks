import torch
import clip
from PIL import Image
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, ContextTypes
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import os

# Настройка логгирования
logging.basicConfig(level=logging.INFO)

# Токен Telegram
TOKEN = 'токен не добавлен для сохранения конфидециальности телеграм-бота'

# Загрузка модели классификации и CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
save_path = " ... " #сюда необходимо вставить путь до одной из папок с весами модели - tripadvisor_hotel_model, либо upbringing_babymodel.
model_classification = BertForSequenceClassification.from_pretrained(save_path)
tokenizer = BertTokenizer.from_pretrained(save_path)
model_classification.to(device)
model_classification.eval()

# Состояния для ConversationHandler
GEOLOCATION, BUDGET, COMPETITORS, RECOMMENDATIONS, STYLE, PHOTOS, CARD, CORRECTION, REVIEW_TEXT = range(9)

# Заранее определенные метки для анализа изображений отеля
labels = [
    "чистый интерьер отеля",
    "чистый экстерьер отеля",
    "ржавчина на сантехнике",
    "низкое качество",
    "плохое освещение",
    "профессиональная фотография"
]
text_inputs = clip.tokenize(labels).to(device)


# Функция для предсказания рейтинга отзыва
def predict_review_rating(review_text):
    inputs = tokenizer(review_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model_classification(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    rating = prediction + 1
    return rating


# Приветствие
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Здравствуйте! Я ваш интеллектуальный ассистент для создания карточки отеля. Введите /help для получения списка команд."
    )


# Список команд и их описание
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Список доступных команд:\n"
        "/start - Приветствие и краткое описание возможностей бота\n"
        "/help - Показать этот список команд\n"
        "/begin - Начать создание карточки отеля\n"
        "/rate_review - Предсказать рейтинг отзыва по его тексту\n"
        "/cancel - Отменить текущий процесс\n"
    )


# Начало работы с карточкой отеля
async def begin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Давайте начнем с ввода геолокации вашего отеля (координаты или адрес).")
    return GEOLOCATION


# Обработка геолокации
async def get_geolocation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    geolocation = update.message.text
    await update.message.reply_text(f"Геолокация принята: {geolocation}. Введите средний бюджет ваших клиентов.")
    return BUDGET


# Бюджет клиентов
async def get_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    budget = update.message.text
    await update.message.reply_text(f"Бюджет принят: {budget}. Пожалуйста, подождите, мы анализируем конкурентов.")
    return COMPETITORS


# Анализ конкурентов
async def analyze_competitors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Ваш главный конкурент определен. Хотите рекомендации по услугам и акциям? Введите 'да' или 'нет'.")
    return RECOMMENDATIONS


# Рекомендации
async def get_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text.lower() == 'да':
        await update.message.reply_text(
            "Рекомендации готовы. Теперь выберите стиль описания: Разговорный / Научный / Публицистический / Официально-Деловой.")
    else:
        await update.message.reply_text("Переход к выбору стиля описания.")
    return STYLE


# Выбор стиля описания
async def choose_style(update: Update, context: ContextTypes.DEFAULT_TYPE):
    style = update.message.text
    await update.message.reply_text(
        f"Стиль описания выбран: {style}. Пожалуйста, отправьте фотографии внутренней и наружной части отеля.")
    return PHOTOS


# Анализ фотографии с помощью CLIP
async def analyze_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    photo_path = "temp.jpg"
    await photo_file.download_to_drive(photo_path)

    image = preprocess(Image.open(photo_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        # Получаем сходства изображений и текстовых меток
        image_features = model_clip.encode_image(image)
        text_features = model_clip.encode_text(text_inputs)

        # Считаем косинусное сходство
        similarity = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

    # Определение метки с наибольшей вероятностью
    top_label = labels[similarity.argmax()]

    # Рекомендации по фото на основе топовой метки
    if top_label == "чистый интерьер отеля":
        await update.message.reply_text("Это фото подходит как изображение интерьера отеля.")
    elif top_label == "чистый экстерьер отеля":
        await update.message.reply_text("Это фото подходит как изображение экстерьера отеля.")
    elif top_label == "ржавчина на сантехнике":
        await update.message.reply_text("На фото видны следы ржавчины. Рекомендуем заменить это изображение.")
    elif top_label == "низкое качество":
        await update.message.reply_text(
            "Фото имеет низкое качество. Возможно, стоит выбрать более качественное изображение.")
    elif top_label == "плохое освещение":
        await update.message.reply_text("Освещение на этом фото слабое. Рекомендуем выбрать более светлое изображение.")
    elif top_label == "профессиональная фотография":
        await update.message.reply_text("Это профессиональное фото. Отличный выбор для карточки отеля.")
    else:
        await update.message.reply_text("Фото проанализировано, но не удалось найти подходящую категорию.")


# Запрос фотографий
async def get_photos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, отправьте по две фотографии интерьера и экстерьера отеля.")
    return PHOTOS


# Генерация карточки отеля
async def generate_card(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Карточка отеля готова. Хотите внести корректировки? Введите 'да' или 'нет'.")
    return CORRECTION


# Корректировка карточки отеля
async def make_correction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text.lower() == 'да':
        await update.message.reply_text("Пожалуйста, введите ваши корректировки.")
    else:
        await update.message.reply_text("Карточка отеля завершена!")
    return ConversationHandler.END


# Тестовая команда для предсказания рейтинга отзыва
async def start_rating(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пожалуйста, отправьте текст отзыва для предсказания рейтинга.")
    return REVIEW_TEXT


# Получение текста отзыва и предсказание рейтинга
async def get_review_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    review_text = update.message.text
    rating = predict_review_rating(review_text)
    await update.message.reply_text(f"Предсказанный рейтинг для этого отзыва: {rating} из 10")
    return ConversationHandler.END


# Завершение работы
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Процесс отменен.")
    return ConversationHandler.END


def main():
    # Инициализация приложения
    application = Application.builder().token(TOKEN).build()

    # Основной обработчик для создания карточки отеля
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('begin', begin)],
        states={
            GEOLOCATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_geolocation)],
            BUDGET: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_budget)],
            COMPETITORS: [MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_competitors)],
            RECOMMENDATIONS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_recommendations)],
            STYLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, choose_style)],
            PHOTOS: [MessageHandler(filters.PHOTO, analyze_photo)],
            CARD: [MessageHandler(filters.TEXT & ~filters.COMMAND, generate_card)],
            CORRECTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, make_correction)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Обработчик диалога для теста предсказания рейтинга отзыва
    review_conv_handler = ConversationHandler(
        entry_points=[CommandHandler('rate_review', start_rating)],
        states={
            REVIEW_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_review_text)],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    # Регистрация обработчиков
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('help', help_command))
    application.add_handler(conv_handler)
    application.add_handler(review_conv_handler)

    # Запуск бота
    application.run_polling()


if __name__ == '__main__':
    main()
