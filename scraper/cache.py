from functools import lru_cache
import hashlib
import json
from datetime import datetime, timedelta

class SearchCache:
    def __init__(self, ttl_hours=24):
        self.cache = {}
        self.ttl = timedelta(hours=ttl_hours)
    
    def get(self, query: str):
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return result
        return None
    
    def set(self, query: str, result):
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = (result, datetime.now())

# Use in agent
cache = SearchCache(ttl_hours=24)

def cached_web_search(query: str):
    cached_result = cache.get(query)
    if cached_result:
        return cached_result
    
    result = web_search_tool.run(query)
    cache.set(query, result)
    return result