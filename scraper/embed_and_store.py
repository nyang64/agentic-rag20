# embed_and_store.py
from scrapy.crawler import CrawlerProcess
from scraper.spiders.recursive_spider import RecursiveSpider
from scrapy.utils.project import get_project_settings
import sys

##
#.  Note:  running embed_and_store.py from one dir above this src file:
#
#.  export SCRAPY_SETTINGS_MODULE=scraper.settings
#.  python -m scraper.embed_and_store https://en.wikivoyage.org/wiki/Driving_in_China
#
# we can hard code settings here: process = CrawlerProcess(settings_dict)
# settings_dict = {
#     "ITEM_PIPELINES": {
#         "pipelines.PgVectorPipeline": 300,
#     },
#     "PGVECTOR_DB_URL": "postgresql://myuser:mypassword@localhost:5432/myprojdb",
#     "HTTPCACHE_ENABLED": False,
#     "DEPTH_LIMIT": 1,
#     "CLOSESPIDER_PAGECOUNT": 500,
#     "DOWNLOAD_DELAY": 0.5,
#     "LOG_LEVEL": "INFO",
# }

if __name__ == "__main__":    
    settings = get_project_settings()
    
    for k in settings.attributes.keys():
        print(k, '->' ,settings[k])
        
    print("  DB URL:", settings["PGVECTOR_DB_URL"])
    
    if len(sys.argv) == 2:
        start_url = sys.argv[1]
    else: 
        start_url = input("\nEnter URL: ").strip()
        #start_url="https://en.wikivoyage.org/wiki/Driving_in_China"
    if not start_url.startswith("http"):
        start_url = "https://" + start_url

    process = CrawlerProcess(settings)  # Uses our settings
    
    process.crawl(RecursiveSpider, start_url=start_url)
    process.start()
