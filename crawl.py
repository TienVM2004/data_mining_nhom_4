from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
import requests
import pandas as pd
import numpy as np
import csv
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# 
# Khai báo hằng
#
NPR_SPORTS_LINK = 'https://www.npr.org/sections/sports/'
NPR_POLITICS_LINK = 'https://www.npr.org/sections/politics/'
NPR_HEALTH_LINK = 'https://www.npr.org/sections/health/'
NPR_SCIENCE_LINK = 'https://www.npr.org/sections/science/'
NPR_BUSINESS_LINK = 'https://www.npr.org/sections/business/'
ALL_CATEGORIES_LINKS = [NPR_POLITICS_LINK, NPR_BUSINESS_LINK, NPR_HEALTH_LINK, NPR_SPORTS_LINK, NPR_SCIENCE_LINK]
LOAD_MORE_BUTTON_CSS_CLASS = 'options__load-more'
LOAD_MORE_BUTTON_XPATH = '//button[@class="options__load-more"]'
TITLE_CLASS = "storytitle"
CONTENT_CLASS = "storytext storylocation linkLocation"
categories = ['politics', 'business', 'health', 'sports', 'science']

#
# Mở dụng Webdriver - Modzilla Firefox
#
def open_web_driver():
    options = Options()
    webdriver_service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(service=webdriver_service, options=options)
    return driver
 #
 # Đóng web driver
 #
def close_web_driver(driver):
    driver.close()
    
#
# Khai báo biến:
#  all_links: Mảng 2 chiều, gồm 5 cột cho 5 categories, mỗi cột chứa n link bài báo
#  driver: Dùng để crawl
#
all_links = []
driver = open_web_driver()


def get_all_links():
    BUTTON_PRESS_TIMES = 50 # Số lần nhấn nút Load more trên web news để có thêm links
    for category in ALL_CATEGORIES_LINKS:
        print(category)
        
        driver.get(category)
        
        time.sleep(5)
        
        # Đối với lần đầu truy cập, web sẽ hiện pop-up và confirm cookies, code dưới đây để tắt cả 2
        if category == ALL_CATEGORIES_LINKS[0]:
            consent_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "#onetrust-accept-btn-handler"))
            )
            consent_button.click()
            time.sleep(5)  # wait for the banner to disappear
                
            button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, ".options.has-more-results .options__load-more"))
                    )
            button.click()
            time.sleep(10)
            
            actions = ActionChains(driver)
            actions.move_by_offset(0, 0)  
            actions.click()
            actions.perform()
            time.sleep(5)
                
        for i in range(BUTTON_PRESS_TIMES):
            print(f"Iteration number {i}")
            try:
                button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".options.has-more-results .options__load-more"))
                )
                button.click()
            except Exception as e:
                print("Something went wrong pressing the button")
                print(e)
                break  
        
        # Biến elements chứa các tag gắn link báo 
        elements = driver.find_elements("xpath", '//a[@data-metrics=\'{"action":"Click Story Title","category":"Story List"}\']')

        links = [element.get_attribute('href') for element in elements]
        all_links.append(links)
      
        print("curently captured a total of ", len(all_links), " links")
        
# Bắt đầu crawl        
get_all_links()
close_web_driver(driver)
#
# Expected output: all_links chứa 5 elements, mỗi elements chứa rất nhiều link báo

# Lưu file link vào để đánh dấu tiến độ
transposed_data = list(map(list, zip(*all_links)))

csv_filename = 'npr_news.csv'

with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    csv_writer.writerow(["politics", "business", "health", "sports", "science"])

    csv_writer.writerows(transposed_data)

print(f"Data extracted and saved to '{csv_filename}'")

#
# Bắt đầu crawl nội dung từng link
#
driver = open_web_driver()

for i in range (len(all_links)):
    data = []
    CSV_SAVE_NAME = categories[i] + '_npr_news.csv'
    with open(CSV_SAVE_NAME, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["title", "content", "category", "link"])
        j = 0
        for link in (all_links[i][:]):
            
            print(link)
            
            title = '' # khởi tạo title và content rỗng, nếu không tìm thấy title và content của báo thì sẽ lưu rỗng vào file data
            content = ''
            
            
            
            driver.get(link)
            html_content = driver.page_source
            soup = BeautifulSoup(html_content, 'html.parser')
            
            titles = soup.find_all('div', class_= TITLE_CLASS)
            contents = soup.find_all('div', class_= CONTENT_CLASS)
            
            if title is not None:
                title = ' '.join(element.text.strip() for element in titles)
                print(f"news number {j} - {categories[i]} title found")
            if content is not None:
                content = ' '.join(element.text.strip() for element in contents)
                content = ' '.join(content.split())
                print(f"news number {j} - {categories[i]} content found")
            # print(title, content)
            data.append([title, content, categories[i], link])
            print(f"news number {j} - {categories[i]} done deal")
            j += 1
        csv_writer.writerows(data)
#
# Expected output: 5 file csv lưu title & content của nhiều bài báo
#
