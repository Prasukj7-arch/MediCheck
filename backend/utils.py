from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_nhs_condition_urls():
    """Scrape NHS A-Z condition page and return a list of condition URLs."""
    base_url = "https://www.nhs.uk"
    index_url = f"{base_url}/conditions/"

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(index_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Correct selector for current NHS structure
        condition_links = soup.select("ul.nhsuk-list--border li a")
        urls = [base_url + link.get("href") for link in condition_links if link.get("href")]
        print(f"✅ Found {len(urls)} condition URLs.")
        return urls
    finally:
        driver.quit()

def scrape_website(url):
    """Scrape text content from a given URL using Selenium."""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # Extract text from relevant tags (e.g., p, h1-h6)
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        text = ' '.join([element.get_text().strip() for element in text_elements])
        # Clean text: remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    finally:
        driver.quit()

def chunk_text(text):
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks