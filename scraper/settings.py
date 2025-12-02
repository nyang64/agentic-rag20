
PROJECT_NAME = "scraper"

BOT_NAME = "scraper"
SPIDER_MODULES = ["scraper.spiders"]
NEWSPIDER_MODULE = "scraper.spiders"

ITEM_PIPELINES = {
    "scraper.pipelines.PgVectorPipeline": 300,
}

PGVECTOR_DB_URL = "postgresql://myuser:mypassword@localhost:5432/myprojdb"

HTTPCACHE_ENABLED = False
DEPTH_LIMIT = 1
CLOSESPIDER_PAGECOUNT = 500
DOWNLOAD_DELAY = 0.5
LOG_LEVEL = "INFO"
ROBOTSTXT_OBEY = True