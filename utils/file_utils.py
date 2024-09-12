import fitz
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path):
    text_data = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                text_data.append({'page': page_num, 'text': text})
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text_data

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text_data = file.read()
    return text_data


def extract_folder_content(folder_path):
    folder_data = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_extension = os.path.splitext(file)[1].lower()
            
            if file_extension == '.pdf':
                pdf_content = extract_text_from_pdf(file_path)
                folder_data.extend(pdf_content)
            elif file_extension in ['.txt', '.md', '.rst']:
                text_content = read_text_file(file_path)
                relative_path = os.path.relpath(file_path, folder_path)
                folder_data.append({'path': relative_path, 'text': text_content})
    print(folder_data)
    return folder_data


# from urllib.parse import urlparse
# import requests
# from bs4 import BeautifulSoup


# def extract_text_from_url(url, timeout=30):
#     logger.debug(f"Attempting to extract text from URL: {url}")
#     try:
#         parsed_url = urlparse(url)
#         if parsed_url.hostname in ['localhost', '127.0.0.1'] or parsed_url.hostname.startswith('192.168.'):
#             if not url.startswith(('http://', 'https://')):
#                 url = 'http://' + url

#         response = requests.get(url, timeout=timeout)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.text, 'html.parser')

#         # Extract text and maintain a hierarchical structure
#         web_data = {
#             'url': url,
#             'title': soup.title.string if soup.title else '',
#             'sections': []
#         }

#         current_section = None
#         for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
#             if element.name in ['h1', 'h2', 'h3']:
#                 if current_section:
#                     web_data['sections'].append(current_section)
#                 current_section = {
#                     'heading': element.get_text(strip=True),
#                     'content': []
#                 }
#             elif element.name in ['p', 'li']:
#                 if current_section:
#                     current_section['content'].append(element.get_text(strip=True))
#                 else:
#                     # If there's content before any heading, create a default section
#                     current_section = {
#                         'heading': 'Introduction',
#                         'content': [element.get_text(strip=True)]
#                     }

#         # Add the last section if it exists
#         if current_section:
#             web_data['sections'].append(current_section)

#         logger.debug(f"Successfully extracted text from URL: {url}")
#         return web_data
#     except requests.RequestException as e:
#         logger.error(f"RequestException error extracting text from URL {url}: {e}")
#         return [{'tag': 'error', 'text': f"Error fetching URL: {str(e)}"}]
#     except Exception as e:
#         logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
#         return [{'tag': 'error', 'text': f"Unexpected error: {str(e)}"}]




import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import logging

logger = logging.getLogger(__name__)

class ImprovedWebScraper:
    def __init__(self, max_pages=10):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.max_pages = max_pages
        self.visited_urls = set()

    def scrape_url(self, url, timeout=30):
        logger.debug(f"Attempting to extract text from URL: {url}")
        try:
            parsed_url = urlparse(url)
            if parsed_url.hostname in ['localhost', '127.0.0.1'] or parsed_url.hostname.startswith('192.168.'):
                if not url.startswith(('http://', 'https://')):
                    url = 'http://' + url
            
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            web_data = {
                'url': url,
                'title': soup.title.string if soup.title else '',
                'sections': []
            }
            
            current_section = None
            for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
                if element.name in ['h1', 'h2', 'h3']:
                    if current_section:
                        web_data['sections'].append(current_section)
                    current_section = {
                        'heading': element.get_text(strip=True),
                        'content': []
                    }
                elif element.name in ['p', 'li']:
                    if current_section:
                        current_section['content'].append(element.get_text(strip=True))
                    else:
                        current_section = {
                            'heading': 'Introduction',
                            'content': [element.get_text(strip=True)]
                        }
            
            if current_section:
                web_data['sections'].append(current_section)
            
            logger.debug(f"Successfully extracted text from URL: {url}")
            return web_data, soup
        
        except requests.RequestException as e:
            logger.error(f"RequestException error extracting text from URL {url}: {e}")
            return [{'tag': 'error', 'text': f"Error fetching URL: {str(e)}"}], None
        except Exception as e:
            logger.error(f"Unexpected error processing URL {url}: {e}", exc_info=True)
            return [{'tag': 'error', 'text': f"Unexpected error: {str(e)}"}], None

    def crawl_website(self, start_url, timeout=30):
        base_url = urlparse(start_url).scheme + "://" + urlparse(start_url).netloc
        to_visit = [start_url]
        all_data = []

        while to_visit and len(self.visited_urls) < self.max_pages:
            url = to_visit.pop(0)
            if url in self.visited_urls:
                continue

            self.visited_urls.add(url)
            web_data, soup = self.scrape_url(url, timeout)

            if isinstance(web_data, list) and web_data[0].get('tag') == 'error':
                all_data.append(web_data[0])
                continue

            all_data.append(web_data)

            if soup:
                for link in soup.find_all('a', href=True):
                    new_url = urljoin(base_url, link['href'])
                    if new_url.startswith(base_url) and new_url not in self.visited_urls:
                        to_visit.append(new_url)

        return all_data

def extract_text_from_url(url, timeout=30, max_pages=10):
    scraper = ImprovedWebScraper(max_pages=max_pages)
    return scraper.crawl_website(url, timeout)