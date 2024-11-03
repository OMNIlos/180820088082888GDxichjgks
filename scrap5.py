import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re
import time

# URL для поиска отелей в Лондоне
search_url = 'https://www.booking.com/searchresults.ru.html?ss=%D0%9B%D0%BE%D0%BD%D0%B4%D0%BE%D0%BD&ssne=%D0%9B%D0%BE%D0%BD%D0%B4%D0%BE%D0%BD&ssne_untouched=%D0%9B%D0%BE%D0%BD%D0%B4%D0%BE%D0%BD&efdco=1&label=nl-nl-booking-desktop-E6jBZCEiIm7Ifcm1lSb5hwS652796017383%3Apl%3Ata%3Ap1%3Ap2%3Aac%3Aap%3Aneg%3Afi%3Atikwd-65526620%3Alp9064730%3Ali%3Adec%3Adm&aid=2311236&lang=ru&sb=1&src_elem=sb&src=searchresults&dest_id=-2601889&dest_type=city&checkin=2024-11-19&checkout=2024-11-22&group_adults=2&no_rooms=1&group_children=0&flex_window=7'

# Запуск драйвера
driver = webdriver.Chrome()
driver.get(search_url)
time.sleep(3)

# Список для хранения ссылок на все отели
hotel_links = []

# Сбор всех ссылок на отели
while True:
    elements = driver.find_elements(By.CSS_SELECTOR, '[data-testid="property-card"]')

    for element in elements:
        link = element.find_element(By.CSS_SELECTOR, '[data-testid="title-link"]').get_attribute('href')
        if link not in hotel_links:
            hotel_links.append(link)

    # Нажимаем на кнопку "Загрузить больше результатов", если доступна
    try:
        load_more_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, 'button.a83ed08757.c21c56c305.bf0537ecb5.f671049264.af7297d90d.c0e0affd09'))
        )
        driver.execute_script("arguments[0].click();", load_more_button)
        time.sleep(3)
    except:
        print("Больше результатов не загружается, сбор ссылок завершен.")
        break

print(f"Собрано ссылок на отели: {len(hotel_links)}")


# Функция для сбора отзывов с каждой страницы отеля
def collect_reviews(hotel_url):
    driver.get(hotel_url)
    time.sleep(3)

    # Открываем вкладку с отзывами
    try:
        elem = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, '[data-testid="Property-Header-Nav-Tab-Trigger-reviews"]'))
        )
        elem.click()
    except:
        print(f"Не удалось найти вкладку с отзывами для {hotel_url}")
        return []
    # Список для хранения всех отзывов с текущего отеля
    all_reviews = []

    while True:
        review_blocks = driver.find_elements(By.CSS_SELECTOR, '[data-testid="review-card"]')

        for review in review_blocks:
            # Сбор оценок
            score_element = review.find_element(By.CSS_SELECTOR, '[data-testid="review-score"]')
            score_text = re.search(r'\d+[,\.]?\d*', score_element.text)
            score = float(score_text.group(0).replace(',', '.')) if score_text else None

            # Сбор положительного отзыва
            try:
                positive_review = review.find_element(By.CSS_SELECTOR, '[data-testid="review-positive-text"]').text
            except:
                positive_review = None  # Если нет положительного отзыва

            # Сбор отрицательного отзыва
            try:
                negative_review = review.find_element(By.CSS_SELECTOR, '[data-testid="review-negative-text"]').text
            except:
                # Если нет отрицательного отзыва
                negative_review = None

            # Проверка на наличия хотя бы одного текста перед добавлением в список
            if positive_review or negative_review:
                all_reviews.append({
                    "score": int(score) if score and score.is_integer() else score,
                    "positive_review": positive_review if positive_review else "Положительный отзыв отсутствует.",
                    "negative_review": negative_review if negative_review else "Отрицательный отзыв отсутствует."
                })

            # Проверяем, достигнут ли лимит отзывов
            if len(all_reviews) >= 50000:
                return all_reviews

        # Переход на следующую страницу с отзывами
        try:
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, 'button[aria-label="Следующая страница"]'))
            )
            driver.execute_script("arguments[0].scrollIntoView();", next_button)
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(3)
        except:
            print("Последняя страница с отзывами достигнута.")
            break

    return all_reviews


# Открытие CSV файла для записи данных
with open("hotel_reviews.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Оценка", "Положительный отзыв", "Отрицательный отзыв"])  # Заголовки столбцов

    collected_reviews = 0
    # Переход по каждой ссылке на отель и сбор отзывов
    for hotel_link in hotel_links:
        if collected_reviews >= 50000:
            # Останавливаем сбор при достижении лимита
            break

        print(f"\nСбор отзывов для: {hotel_link}")
        reviews = collect_reviews(hotel_link)

        for review in reviews:
            writer.writerow([review['score'], review['positive_review'], review['negative_review']])
            collected_reviews += 1

            if collected_reviews >= 50000:
                # Останавливаем сбор, если достигнут лимит отзывов
                break

driver.quit()
