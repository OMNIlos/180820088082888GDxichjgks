import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# Обновленные заголовки для обхода ошибки 401
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Connection": "keep-alive",
    "Accept-Encoding": "gzip, deflate, br",
    "Cookie": "ma_id=2712419181726595939132; dspid=9d495323-ef6d-47b7-a9e8-0bc3fbe5a38b; reset_cookie=1; trvUserId=7f437864-c906-4491-bb56-42b89304ff1c; _ymab_param=eJ8ghNClaZIaocUP5y1CGZRlV9zZyGlWX-iSUUOZwcbj8rH8PZQurYMUR8VujipmNAlv5Z6bazUAwOXraftrjPviZzA; _gcl_au=1.1.291971588.1729879715; ma_cid=5682801121729879715; mcid=19309062001729879715; first_hit_timestamp=1729879715071; st_uid=979f9abb5d6c77d6814b4670f766c6a4; _ga=GA1.2.767968362.1729879715; _ym_uid=1729879715763558871; _ym_d=1729879715; tmr_lvid=dbee84a70f6339dbcc8bf7dfffcdcc4a; tmr_lvidTS=1729879715365; ma_prevVisId_2392180003=053e5a62084607ccc243b0b29d7e7e6f; mts_id=5f7f52bc-00c9-456d-9f2b-91f6e42ae29e; adrcid=A4W7ijHjYeIT03uof2Y-n2w; adrcid=A4W7ijHjYeIT03uof2Y-n2w; _pa_aaid=848074db-8a30-4b4a-946f-bd75d073eb1e; PARTNERSHIP_PORTAL_SESSION_ID=OGI4ZDBkODEtMWVmZS00ZTY1LTg3MjQtMjg2NmE3Y2NhNzQy; TS0121a975=012019f3d460e1d4bdb3833ea8826ea37ccba41db76b3cfa013ee5b0b9e7a1a618ebf871316e6174cf78abb409cd14648245893473af5b4243bbb2667a457211430c2f25de42076bf0b436553ed4d0e4d4f1c01468; mcid=960047741729928785; first_hit_timestamp=1729928785783; cd10=15; _ga=GA1.1.344684988.1729946897; _ga_H2MY6J9ZKV=GS1.1.1729946896.1.1.1729946898.58.0.0; banners-set-3=39; cd10=16; mvid=19309062001729879715_1729879715071; mts_id_last_sync=1730050004; ma_prevFp_2392180003=2258264886|3665706458|430672962|888000370|749129358|959198862|3530670207|888000370|1068634537|3180462103|1068634537|2232987877|854988931|3579944471|3180462103|888000370|888000370|1068634537|3539069274|888000370|621576841|3530670207|3579944471|1068634537|3191396633|668684770|3579944471|3164198037|2360593094|888000370|2895824364|3708322660|718548439|3308070491; ma_last_sync=1730050025575; _ym_isad=2; ma_id_api=iaTHpddpyD07U8c4qnchWZ6BypcnCHAbYFScNYINSEF34QRDN0Lne2bCuHecISKK7mdXDT1yfdxlEPWScHiUfMIoWXM3JXtzyMX+FkoO8aTjCElvyNHsn521PLKC8JLn0TmhH4sfeY23HzD1GGbeWHz8RFVXqcznN7TZw7QNNh0QTf+4PZo9+Fg5eWXyKEKWrEfpxGNlIX5pz2lAagfc3lwC6CtH5kQmvNI7grfFNZFmwsY71DzjoXev85TU86DnBp6zRgGtEbZrlEvjptWym6rGtN2xYDMiUa1lNIa1vTofeMt5C2E/7BUyW5khNNRdKp/6DKWJQkwnScGx45fYJA==; ma_vis_id_last_sync_2392180003=1730050025761; _ym_visorc=w; adrdel=1730050026655; adrdel=1730050026655; _ymab_flags_={%22i%22:%22eJ8ghNClaZIaocUP5y1CGZRlV9zZyGlWX-iSUUOZwcbj8rH8PZQurYMUR8VujipmNAlv5Z6bazUAwOXraftrjPviZzA%22%2C%22experiments%22:%22KeTYCAoAP6KKJgO-3p4MoQ%2C%2C%22%2C%22flags%22:{%22CARD_WITH_PAYMENT%22:%22true%22}}; qrator_jsid2=v2.0.1730050021.098.af6e7233qozR9Kl0|HtmPpMizJi6oBFmo|YTkcGLE0ly/r9Rx3RUOFxEKknC9SIVJOcEqo1O1V4C84Wlw+2e3vAs97lAcrkraIgLt+V8V8W/LCzlxUkUg6atwXfK9DaLvpXhbHZBVwkM0qUuqTCk8xH7ba4Kq0RiGOq7PhX91mjpHlGg2ywVN1eOvmdCh6ackelAXcP2INdK0=-iOeWDkn6TH1nsxF2BH9qUwVg7b0=; banners-set-2=61; _gcl_gs=2.1.k1$i1730051940$u23159167; _gcl_aw=GCL.1730051942.Cj0KCQjwpvK4BhDUARIsADHt9sTOhy6PpLQeNZYFxIii3LykkUJdg9gmw2r5X7aSugx1di5kh9I9d5kaAufbEALw_wcB; tmr_detect=0%7C1730051946392; ma_ss_300a449b-3cb5-4ae4-b212-06691be833f2=8077483621730051560.10.1730051954.50"
}


# Шаг 1: Сбор ссылок на страницы стран с главной страницы
def scrape_country_links(url, headers):
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка запроса на главной странице: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    country_links = []

    # Извлечение ссылок на страницы стран
    for link in soup.select('a.flex.w-full.flex-col.gap-2'):
        if 'href' in link.attrs:
            country_links.append(f"https://travel.mts.ru{link['href']}")

    return country_links


# Шаг 2: Сбор ссылок на отели на странице страны
def scrape_hotel_links(country_url, headers):
    response = requests.get(country_url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка запроса на странице страны: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    hotel_links = []

    # Извлечение ссылок на страницы отелей
    for hotel_card in soup.select('div[data-test="hotel_card"]'):
        link_tag = hotel_card.find('a', href=True)
        if link_tag:
            hotel_links.append(f"https://travel.mts.ru{link_tag['href']}")

    return hotel_links


# Шаг 3: Сбор отзывов с одной страницы отеля
def scrape_hotel_reviews(hotel_url, headers):
    response = requests.get(hotel_url, headers=headers)
    if response.status_code != 200:
        print(f"Ошибка запроса на странице отеля: {response.status_code}")
        return pd.DataFrame()

    soup = BeautifulSoup(response.text, 'html.parser')
    reviews = []

    review_blocks = soup.select('div.rounded-3xl.border')  # Оберточный блок для отзывов

    for review in review_blocks:
        # Извлечение текста отзыва
        text = review.select_one('p.p2-regular.text-primary.line-clamp-3').get_text(strip=True) if review.select_one(
            'p.p2-regular.text-primary.line-clamp-3') else ""

        # Извлечение автора
        author = review.select_one('p.mtsds-p3-regular-comp.text-primary[data-test="review_author"]').get_text(
            strip=True) if review.select_one('p.mtsds-p3-regular-comp.text-primary[data-test="review_author"]') else ""

        # Извлечение даты
        date = review.select_one('p.mtsds-p4-regular-comp.text-secondary[data-test="review_date"]').get_text(
            strip=True) if review.select_one('p.mtsds-p4-regular-comp.text-secondary[data-test="review_date"]') else ""

        # Извлечение рейтинга
        rating = review.select_one('div.flex.size-full.items-center.justify-center.gap-0\\.5').get_text(
            strip=True) if review.select_one('div.flex.size-full.items-center.justify-center.gap-0\\.5') else ""

        # Добавление данных
        reviews.append({
            'author': author,
            'date': date,
            'rating': rating,
            'text': text
        })

    return pd.DataFrame(reviews)


# Шаг 4: Основная функция для сбора отзывов со всех отелей на всех страницах стран
def scrape_all_hotels(url, headers):
    country_links = scrape_country_links(url, headers)  # Получаем ссылки на все страны
    all_reviews = []

    for country_link in country_links:
        hotel_links = scrape_hotel_links(country_link, headers)  # Ссылки на отели для каждой страны

        for hotel_link in hotel_links:
            reviews = scrape_hotel_reviews(hotel_link, headers)
            if not reviews.empty:
                all_reviews.append(reviews)
                print(f"Собрано {len(reviews)} отзывов с {hotel_link}")

            time.sleep(1)  # Пауза для предотвращения блокировки

    # Проверка, есть ли данные для объединения
    if all_reviews:
        all_reviews_df = pd.concat(all_reviews, ignore_index=True)
        return all_reviews_df
    else:
        print("Нет данных для объединения.")
        return pd.DataFrame()


# Главный URL сайта и запуск сбора данных
main_url = "https://travel.mts.ru"
all_reviews_df = scrape_all_hotels(main_url, headers)

# Сохранение всех данных в CSV-файл, если данные существуют
if not all_reviews_df.empty:
    all_reviews_df.to_csv('all_hotel_reviews.csv', index=False)
    print("Данные успешно сохранены в all_hotel_reviews.csv")
else:
    print("Сбор данных завершен, но нет данных для сохранения.")

