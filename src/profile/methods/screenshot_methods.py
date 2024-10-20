import os

from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from io import BytesIO

# screenshot -> analyzing
def selenium_screenshot(url):
    driver = webdriver.Chrome(service= Service(ChromeDriverManager().install()))
    driver.get(url=url)
    
    sleep(2)
    
    driver.execute_script("document.body.style.zoom='80%'")
    screenshot = driver.get_screenshot_as_base64()
    
    sleep(2)
    
    driver.quit()
    return screenshot