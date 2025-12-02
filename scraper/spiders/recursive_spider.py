# scraper/spiders/recursive_spider.py
import scrapy
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup


class PageItem(scrapy.Item):
    url = scrapy.Field()
    title = scrapy.Field()
    text = scrapy.Field()


class RecursiveSpider(scrapy.Spider):
    name = "recursive"
    custom_settings = {
        #"DOWNLOAD_DELAY": 1.0,          # be polite
        #"ROBOTSTXT_OBEY": True,
    }

    def __init__(self, start_url=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not start_url:
            raise ValueError("start_url is required")
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]

    def parse(self, response):
        # Extract clean text
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        title = soup.title.string.strip() if soup.title else ""
        text = soup.get_text(separator=" ", strip=True)

        yield PageItem(url=response.url, title=title, text=text)

        # Follow internal links
        for a in soup.find_all("a", href=True):
            link = urljoin(response.url, a["href"])
            parsed = urlparse(link)
            if parsed.netloc == self.allowed_domains[0]:
                yield scrapy.Request(link, callback=self.parse)

