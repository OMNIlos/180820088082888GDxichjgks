# Интеллектуальный ассистент отельера
## Ссылка на открытый гугл диск с весами для моделей : [Weights_folder](https://drive.google.com/drive/folders/1HYi87n9x0Mb3pl7G6O4Tgc0nBbpouW0J?usp=sharing)
1. **Содержимое открытого гугл диска** :
* Файл **trip_advisor_model** содержит в себе веса, токены и.т.д для модели [tripadvisor_model.py](tripadvisor_model.py), дообученной на датасете.
* Файл **upbringing_babymodel** содержит в себе веса, токены и.т.д для модели [Model_Learning.py](Model_Learning.py), дообученной на собранных методом веб-скрайпинга данных.
## Содержание репозитория :
### Структура файлов : 
1. **Данные с которыми мы работали распределены по файлам следующим образом** :
* [hotel_reviews.csv](hotel_reviews.csv) - в этом файле находится таблица данных, собранных с сайта [booking.com](booking.com) с помошью парсинга.
* [cleaned_booking_reviews.csv](cleaned_booking_reviews.csv) - предобработанный [hotel_reviews.csv](hotel_reviews.csv).
* [tripadvisor_hotel_reviews.csv](tripadvisor_hotel_reviews.csv) - датасет скаченный с [kaggle.com](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews), на нем была обучена вторая модель(обучили две одинаковые модели на двух разных датасетах)
2. **Метод сбора данных(мы осуществили два метода сбора данны)** :
* Осушествили парсинг сайта [booking.com](booking.com) с помощью созданного нами скрипта расположенного в файле [Scrap5.py](scrap5.py).
* Подобрали и скачали наилучший для нашей задачи датасет c [kaggle.com](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews) : [hotel_reviews.csv](hotel_reviews.csv).
3. **Исследовательский анализ данных(мы провели исследовательский анализ данных EDA для всех собранных данных)** :
* Исследовательский анализ данных для [cleaned_booking_reviews.csv](cleaned_booking_reviews.csv) расположен в файле [EDA.py](EDA.py).
* Исследовательский анализ данных для [tripadvisor_hotel_reviews.csv](tripadvisor_hotel_reviews.csv) расположен в файле [EDA_for_tripadvisor.py](EDA_for_tripadvisor.py).
4. **Файлы для предобработки данных** :
* Код для предобработки собранных данных [hotel_reviews.csv](hotel_reviews.csv) в данные [cleaned_booking_reviews.csv](cleaned_booking_reviews.csv), расположен в файле [Data_preprocessing.py](Data_preprocessing.py).
* Код для токенизации [cleaned_booking_reviews.csv](cleaned_booking_reviews.csv), расположен в файле [BERT_Data_Tokenization.py](BERT_Data_Tokenization.py).
5. **Файлы с моделями-классификаторами, предсказывающей рейтинг отеля по его отзывам** :
* Модель обученная на датасете [tripadvisor_hotel_reviews.csv](tripadvisor_hotel_reviews.csv), расположена в файле [tripadvisor_model.py](tripadvisor_model.py).
* Модель для дообучения [tripadvisor_model.py](tripadvisor_model.py), пока не создана, т.к. качества работы tripadvisor_model.py для тествого формата достаточно. 
* Модель обученная один раз на датасете [cleaned_booking_reviews.csv](cleaned_booking_reviews.csv), расположена в файле [Model_Learning.py](Model_Learning.py).
* Модель для дообучения [Model_Learning.py](Model_Learning.py), расположена в файле [Model.py](Model.py).
### Все библеотеки для каждого из файлов :
1. Файл Scrap5.py : `pip install csv selenium re time -q`.
2. Файл mtsbot.py : `pip install torch clip PIL telegram transformers logging -q`, также полезной практикой будет импортировать `import os`.
3. Файл tripadvisor_model.py, Model_learning.py, Model.py : `pip install pandas torch scikit-learn signal -q`, также полезной практикой будет импортировать `import os`.
4. Файл Data_preprocessing.py :  `pip install pandas re nltk razdel ssl -q`.
5. Файл EDA.py, EDA_for_tripadvisor.py : `pip install matplotlib wordcloud scikit-learn nltk pandas re -q`.
6. Файл BERT_DATA_Tokenizer.py : `pip install pandas transformers`.
