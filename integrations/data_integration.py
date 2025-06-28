import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
import feedparser
import sqlite3
import aiohttp
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIntegration:
    """Real-time data integration module for the AI Problem Solver.
    
    This module provides capabilities to fetch and process data from various sources:
    - Web APIs (REST, GraphQL)
    - Web scraping
    - Financial data
    - News and RSS feeds
    - Weather data
    - Local databases
    - CSV/Excel files
    
    It includes caching mechanisms to avoid redundant requests and rate limiting
    to respect API usage policies.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, cache_ttl: int = 3600):
        """Initialize the data integration module.
        
        Args:
            cache_dir: Directory to store cached data. If None, a temporary directory will be used.
            cache_ttl: Time-to-live for cached data in seconds (default: 1 hour).
        """
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".ai_problem_solver", "cache")
        self.cache_ttl = cache_ttl
        self.api_rate_limits = {}
        self.api_keys = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize session for HTTP requests
        self.session = requests.Session()
        
        # Load API keys if available
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment variables or configuration file."""
        # Try to load from environment variables
        for key in os.environ:
            if key.endswith('_API_KEY'):
                service = key.replace('_API_KEY', '').lower()
                self.api_keys[service] = os.environ[key]
        
        # Try to load from configuration file
        config_path = os.path.join(os.path.expanduser("~"), ".ai_problem_solver", "config", "api_keys.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    keys = json.load(f)
                    self.api_keys.update(keys)
            except Exception as e:
                logger.warning(f"Could not load API keys from configuration file: {str(e)}")
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cached item.
        
        Args:
            cache_key: Unique identifier for the cached item.
            
        Returns:
            File path for the cached item.
        """
        # Create a safe filename from the cache key
        safe_key = ''.join(c if c.isalnum() else '_' for c in cache_key)
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if a cached item is still valid.
        
        Args:
            cache_path: File path for the cached item.
            
        Returns:
            True if the cached item is still valid, False otherwise.
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check if the cache has expired
        modified_time = os.path.getmtime(cache_path)
        current_time = time.time()
        return (current_time - modified_time) < self.cache_ttl
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available and valid.
        
        Args:
            cache_key: Unique identifier for the cached item.
            
        Returns:
            Cached data if available and valid, None otherwise.
        """
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cached data: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache.
        
        Args:
            cache_key: Unique identifier for the cached item.
            data: Data to cache.
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Could not save data to cache: {str(e)}")
    
    def _check_rate_limit(self, api_name: str, limit_per_minute: int = 60) -> bool:
        """Check if an API request would exceed the rate limit.
        
        Args:
            api_name: Name of the API.
            limit_per_minute: Maximum number of requests allowed per minute.
            
        Returns:
            True if the request is allowed, False if it would exceed the rate limit.
        """
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Initialize or clean up the rate limit history
        if api_name not in self.api_rate_limits:
            self.api_rate_limits[api_name] = []
        
        # Remove requests older than a minute
        self.api_rate_limits[api_name] = [t for t in self.api_rate_limits[api_name] if t > minute_ago]
        
        # Check if the rate limit would be exceeded
        if len(self.api_rate_limits[api_name]) >= limit_per_minute:
            return False
        
        # Add the current request to the history
        self.api_rate_limits[api_name].append(current_time)
        return True
    
    def fetch_rest_api(self, url: str, method: str = 'GET', params: Optional[Dict] = None, 
                      headers: Optional[Dict] = None, data: Optional[Dict] = None, 
                      auth: Optional[Tuple] = None, use_cache: bool = True, 
                      cache_ttl: Optional[int] = None, rate_limit: int = 60) -> Dict:
        """Fetch data from a REST API.
        
        Args:
            url: API endpoint URL.
            method: HTTP method (GET, POST, PUT, DELETE, etc.).
            params: Query parameters.
            headers: HTTP headers.
            data: Request body for POST/PUT requests.
            auth: Authentication tuple (username, password).
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            rate_limit: Maximum number of requests allowed per minute.
            
        Returns:
            API response as a dictionary.
            
        Raises:
            requests.exceptions.RequestException: If the API request fails.
        """
        # Create a cache key based on the request parameters
        cache_key = f"rest_{method}_{url}_{str(params)}_{str(data)}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Check rate limit
        if not self._check_rate_limit(url, rate_limit):
            logger.warning(f"Rate limit exceeded for {url}. Waiting...")
            time.sleep(60 / rate_limit)  # Wait for rate limit to reset
        
        # Make the API request
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                json=data,
                auth=auth
            )
            response.raise_for_status()
            
            # Parse the response
            if response.headers.get('content-type', '').startswith('application/json'):
                result = response.json()
            else:
                result = {"text": response.text, "status_code": response.status_code}
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise
    
    def fetch_graphql(self, url: str, query: str, variables: Optional[Dict] = None, 
                     headers: Optional[Dict] = None, use_cache: bool = True, 
                     cache_ttl: Optional[int] = None, rate_limit: int = 60) -> Dict:
        """Fetch data from a GraphQL API.
        
        Args:
            url: GraphQL endpoint URL.
            query: GraphQL query string.
            variables: GraphQL variables.
            headers: HTTP headers.
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            rate_limit: Maximum number of requests allowed per minute.
            
        Returns:
            GraphQL response as a dictionary.
            
        Raises:
            requests.exceptions.RequestException: If the GraphQL request fails.
        """
        # Create a cache key based on the request parameters
        cache_key = f"graphql_{url}_{query}_{str(variables)}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Check rate limit
        if not self._check_rate_limit(url, rate_limit):
            logger.warning(f"Rate limit exceeded for {url}. Waiting...")
            time.sleep(60 / rate_limit)  # Wait for rate limit to reset
        
        # Prepare the GraphQL request
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        # Make the API request
        try:
            response = self.session.post(
                url=url,
                json=payload,
                headers=headers or {}
            )
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"GraphQL request failed: {str(e)}")
            raise
    
    def scrape_webpage(self, url: str, css_selector: Optional[str] = None, 
                      xpath: Optional[str] = None, use_cache: bool = True, 
                      cache_ttl: Optional[int] = None, rate_limit: int = 10) -> Dict:
        """Scrape data from a webpage.
        
        Args:
            url: URL of the webpage to scrape.
            css_selector: CSS selector to extract specific elements.
            xpath: XPath to extract specific elements.
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            rate_limit: Maximum number of requests allowed per minute.
            
        Returns:
            Dictionary containing the scraped data.
            
        Raises:
            requests.exceptions.RequestException: If the HTTP request fails.
        """
        # Create a cache key based on the request parameters
        cache_key = f"scrape_{url}_{css_selector}_{xpath}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Check rate limit (lower for scraping to be respectful)
        if not self._check_rate_limit(url, rate_limit):
            logger.warning(f"Rate limit exceeded for {url}. Waiting...")
            time.sleep(60 / rate_limit)  # Wait for rate limit to reset
        
        # Make the HTTP request
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract data based on selectors
            result = {
                "url": url,
                "title": soup.title.text if soup.title else "",
                "text": soup.get_text()
            }
            
            if css_selector:
                elements = soup.select(css_selector)
                result["selected_elements"] = [str(el) for el in elements]
                result["selected_text"] = [el.get_text() for el in elements]
            
            if xpath:
                # BeautifulSoup doesn't support XPath directly, so we'd need to use lxml
                # This is a simplified version
                from lxml import etree
                html = etree.HTML(response.text)
                elements = html.xpath(xpath)
                result["xpath_elements"] = [etree.tostring(el).decode() if hasattr(el, 'tag') else str(el) for el in elements]
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, result)
            
            return result
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Web scraping failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            raise
    
    def fetch_stock_data(self, symbols: Union[str, List[str]], period: str = "1mo", 
                        interval: str = "1d", use_cache: bool = True, 
                        cache_ttl: Optional[int] = None) -> Dict:
        """Fetch stock market data.
        
        Args:
            symbols: Stock symbol(s) to fetch data for.
            period: Time period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo).
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            
        Returns:
            Dictionary containing the stock data.
        """
        # Convert single symbol to list
        if isinstance(symbols, str):
            symbols = [symbols]
        
        # Create a cache key based on the request parameters
        cache_key = f"stock_{','.join(symbols)}_{period}_{interval}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Fetch stock data using yfinance
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, interval=interval)
                
                # Convert DataFrame to dictionary
                hist_dict = hist.reset_index().to_dict(orient='records')
                
                # Get additional info
                info = ticker.info
                
                data[symbol] = {
                    "history": hist_dict,
                    "info": info
                }
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, data)
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            raise
    
    def fetch_news(self, query: Optional[str] = None, sources: Optional[List[str]] = None, 
                  category: Optional[str] = None, limit: int = 10, 
                  use_cache: bool = True, cache_ttl: Optional[int] = None) -> List[Dict]:
        """Fetch news articles.
        
        Args:
            query: Search query for news articles.
            sources: List of news sources to fetch from.
            category: News category (business, entertainment, health, science, sports, technology).
            limit: Maximum number of articles to fetch.
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            
        Returns:
            List of news articles.
        """
        # Create a cache key based on the request parameters
        cache_key = f"news_{query}_{','.join(sources or [])}_{category}_{limit}"
        
        # Try to get from cache if enabled
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        # Check if we have a News API key
        if 'newsapi' in self.api_keys:
            return self._fetch_news_api(query, sources, category, limit, cache_key, use_cache)
        else:
            return self._fetch_news_rss(query, sources, limit, cache_key, use_cache)
    
    def _fetch_news_api(self, query: Optional[str], sources: Optional[List[str]], 
                       category: Optional[str], limit: int, 
                       cache_key: str, use_cache: bool) -> List[Dict]:
        """Fetch news using the News API."""
        try:
            # Prepare parameters
            params = {
                "apiKey": self.api_keys['newsapi'],
                "pageSize": limit
            }
            
            if query:
                params["q"] = query
            
            if sources:
                params["sources"] = ",".join(sources)
            
            if category:
                params["category"] = category
            
            # Make the API request
            response = self.session.get("https://newsapi.org/v2/top-headlines", params=params)
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            articles = result.get("articles", [])
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, articles)
            
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching news from News API: {str(e)}")
            # Fall back to RSS feeds
            return self._fetch_news_rss(query, sources, limit, cache_key, use_cache)
    
    def _fetch_news_rss(self, query: Optional[str], sources: Optional[List[str]], 
                       limit: int, cache_key: str, use_cache: bool) -> List[Dict]:
        """Fetch news from RSS feeds."""
        try:
            # Default RSS feeds if none provided
            default_feeds = [
                "http://rss.cnn.com/rss/cnn_topstories.rss",
                "http://feeds.bbci.co.uk/news/rss.xml",
                "http://feeds.reuters.com/reuters/topNews",
                "https://www.theguardian.com/world/rss"
            ]
            
            feeds_to_fetch = sources or default_feeds
            all_entries = []
            
            for feed_url in feeds_to_fetch:
                try:
                    feed = feedparser.parse(feed_url)
                    all_entries.extend(feed.entries)
                except Exception as e:
                    logger.warning(f"Error fetching RSS feed {feed_url}: {str(e)}")
            
            # Filter by query if provided
            if query:
                query = query.lower()
                filtered_entries = []
                for entry in all_entries:
                    title = entry.get('title', '').lower()
                    summary = entry.get('summary', '').lower()
                    if query in title or query in summary:
                        filtered_entries.append(entry)
                all_entries = filtered_entries
            
            # Sort by publication date (newest first)
            all_entries.sort(key=lambda x: x.get('published_parsed', 0), reverse=True)
            
            # Limit the number of entries
            all_entries = all_entries[:limit]
            
            # Convert to a more standardized format
            articles = []
            for entry in all_entries:
                article = {
                    "title": entry.get('title', ''),
                    "description": entry.get('summary', ''),
                    "url": entry.get('link', ''),
                    "publishedAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', entry.get('published_parsed')) if entry.get('published_parsed') else None,
                    "source": {
                        "name": entry.get('source', {}).get('title', 'Unknown')
                    }
                }
                articles.append(article)
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, articles)
            
            return articles
        
        except Exception as e:
            logger.error(f"Error fetching news from RSS feeds: {str(e)}")
            return []
    
    def fetch_weather(self, location: str, units: str = "metric", 
                     use_cache: bool = True, cache_ttl: Optional[int] = None) -> Dict:
        """Fetch weather data for a location.
        
        Args:
            location: Location to fetch weather data for (city name, zip code, coordinates).
            units: Units of measurement (metric, imperial, standard).
            use_cache: Whether to use cached data if available.
            cache_ttl: Time-to-live for cached data in seconds (overrides the default).
            
        Returns:
            Dictionary containing the weather data.
        """
        # Create a cache key based on the request parameters
        cache_key = f"weather_{location}_{units}"
        
        # Try to get from cache if enabled (with shorter TTL for weather data)
        actual_cache_ttl = cache_ttl or min(self.cache_ttl, 3600)  # Max 1 hour for weather
        if use_cache:
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                cache_time = os.path.getmtime(self._get_cache_path(cache_key))
                if (time.time() - cache_time) < actual_cache_ttl:
                    return cached_data
        
        # Check if we have an OpenWeatherMap API key
        if 'openweathermap' in self.api_keys:
            try:
                # Prepare parameters
                params = {
                    "q": location,
                    "appid": self.api_keys['openweathermap'],
                    "units": units
                }
                
                # Make the API request
                response = self.session.get("https://api.openweathermap.org/data/2.5/weather", params=params)
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                
                # Cache the result if enabled
                if use_cache:
                    self._save_to_cache(cache_key, result)
                
                return result
            
            except Exception as e:
                logger.error(f"Error fetching weather data from OpenWeatherMap: {str(e)}")
                # Fall back to web scraping
        
        # Fall back to web scraping if no API key or API request failed
        try:
            # Scrape weather data from a public website
            url = f"https://www.google.com/search?q=weather+{location.replace(' ', '+')}"
            result = self.scrape_webpage(url, use_cache=False)  # Don't cache the intermediate result
            
            # Extract weather information from the scraped data
            # This is a simplified version and might not be reliable
            weather_data = {
                "location": location,
                "source": "web_scraping",
                "raw_text": result.get("text", ""),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result if enabled
            if use_cache:
                self._save_to_cache(cache_key, weather_data)
            
            return weather_data
        
        except Exception as e:
            logger.error(f"Error scraping weather data: {str(e)}")
            return {"error": "Could not fetch weather data", "location": location}
    
    def query_database(self, query: str, params: Optional[List] = None, 
                      db_path: Optional[str] = None, db_type: str = "sqlite") -> List[Dict]:
        """Query a local database.
        
        Args:
            query: SQL query to execute.
            params: Parameters for the SQL query.
            db_path: Path to the database file.
            db_type: Database type (sqlite, mysql, postgresql).
            
        Returns:
            List of query results as dictionaries.
            
        Raises:
            ValueError: If the database type is not supported.
            Exception: If the database query fails.
        """
        if db_type.lower() != "sqlite":
            raise ValueError(f"Database type {db_type} is not supported yet. Only SQLite is supported.")
        
        if not db_path:
            raise ValueError("Database path must be provided for SQLite.")
        
        try:
            # Connect to the database
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row  # This enables column access by name
            cursor = conn.cursor()
            
            # Execute the query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Fetch the results
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            results = [dict(row) for row in rows]
            
            # Close the connection
            conn.close()
            
            return results
        
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            **kwargs: Additional arguments to pass to pandas.read_csv().
            
        Returns:
            Pandas DataFrame containing the CSV data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the file cannot be read.
        """
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_excel(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            sheet_name: Name of the sheet to load.
            **kwargs: Additional arguments to pass to pandas.read_excel().
            
        Returns:
            Pandas DataFrame containing the Excel data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            Exception: If the file cannot be read.
        """
        try:
            return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise
    
    async def fetch_multiple_apis(self, urls: List[str], method: str = 'GET', 
                                headers: Optional[Dict] = None, params: Optional[List[Dict]] = None, 
                                data: Optional[List[Dict]] = None) -> List[Dict]:
        """Fetch data from multiple APIs asynchronously.
        
        Args:
            urls: List of API endpoint URLs.
            method: HTTP method (GET, POST, PUT, DELETE, etc.).
            headers: HTTP headers to use for all requests.
            params: List of query parameters for each request (must match length of urls).
            data: List of request bodies for each request (must match length of urls).
            
        Returns:
            List of API responses as dictionaries.
        """
        async def fetch_url(session, url, method, headers, params, data):
            try:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers, params=params) as response:
                        response.raise_for_status()
                        if response.headers.get('content-type', '').startswith('application/json'):
                            return await response.json()
                        else:
                            return {"text": await response.text(), "status_code": response.status}
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, json=data) as response:
                        response.raise_for_status()
                        if response.headers.get('content-type', '').startswith('application/json'):
                            return await response.json()
                        else:
                            return {"text": await response.text(), "status_code": response.status}
                else:
                    return {"error": f"Method {method} not supported for async requests"}
            except Exception as e:
                return {"error": str(e), "url": url}
        
        # Ensure params and data lists match the length of urls
        if params is None:
            params = [None] * len(urls)
        elif len(params) != len(urls):
            params = params + [None] * (len(urls) - len(params))
        
        if data is None:
            data = [None] * len(urls)
        elif len(data) != len(urls):
            data = data + [None] * (len(urls) - len(data))
        
        # Create a new aiohttp session
        async with aiohttp.ClientSession() as session:
            # Create tasks for all URLs
            tasks = []
            for i, url in enumerate(urls):
                tasks.append(fetch_url(session, url, method, headers, params[i], data[i]))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            return results
    
    def fetch_multiple_apis_sync(self, urls: List[str], method: str = 'GET', 
                               headers: Optional[Dict] = None, params: Optional[List[Dict]] = None, 
                               data: Optional[List[Dict]] = None) -> List[Dict]:
        """Fetch data from multiple APIs synchronously (wrapper for async method).
        
        Args:
            urls: List of API endpoint URLs.
            method: HTTP method (GET, POST, PUT, DELETE, etc.).
            headers: HTTP headers to use for all requests.
            params: List of query parameters for each request (must match length of urls).
            data: List of request bodies for each request (must match length of urls).
            
        Returns:
            List of API responses as dictionaries.
        """
        # Create an event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async method
            results = loop.run_until_complete(
                self.fetch_multiple_apis(urls, method, headers, params, data)
            )
            return results
        finally:
            # Close the event loop
            loop.close()
    
    def analyze_data(self, data: Union[List, Dict, pd.DataFrame], analysis_type: str, 
                    params: Optional[Dict] = None) -> Dict:
        """Analyze data using various statistical methods.
        
        Args:
            data: Data to analyze (list, dictionary, or DataFrame).
            analysis_type: Type of analysis to perform (summary, correlation, regression, etc.).
            params: Additional parameters for the analysis.
            
        Returns:
            Dictionary containing the analysis results.
        """
        # Convert data to DataFrame if it's not already
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                raise ValueError("Data must be a list of dictionaries, a dictionary, or a DataFrame.")
        else:
            df = data
        
        # Perform the requested analysis
        if analysis_type.lower() == "summary":
            return self._analyze_summary(df, params)
        elif analysis_type.lower() == "correlation":
            return self._analyze_correlation(df, params)
        elif analysis_type.lower() == "regression":
            return self._analyze_regression(df, params)
        elif analysis_type.lower() == "time_series":
            return self._analyze_time_series(df, params)
        else:
            raise ValueError(f"Analysis type {analysis_type} is not supported.")
    
    def _analyze_summary(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """Generate summary statistics for a DataFrame."""
        params = params or {}
        
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate summary statistics
        summary = {
            "count": df.shape[0],
            "columns": df.shape[1],
            "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": numeric_df.describe().to_dict() if not numeric_df.empty else {}
        }
        
        # Add categorical summaries if requested
        if params.get("include_categorical", True):
            categorical_df = df.select_dtypes(exclude=[np.number])
            if not categorical_df.empty:
                summary["categorical_summary"] = {}
                for col in categorical_df.columns:
                    summary["categorical_summary"][col] = {
                        "unique_values": categorical_df[col].nunique(),
                        "top_values": categorical_df[col].value_counts().head(5).to_dict()
                    }
        
        return summary
    
    def _analyze_correlation(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """Calculate correlations between columns in a DataFrame."""
        params = params or {}
        method = params.get("method", "pearson")
        min_periods = params.get("min_periods", 1)
        
        # Get numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"error": "No numeric columns found for correlation analysis."}
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr(method=method, min_periods=min_periods)
        
        # Convert to dictionary
        result = {
            "correlation_matrix": corr_matrix.to_dict(),
            "method": method
        }
        
        # Find highest correlations if requested
        if params.get("find_highest", True):
            # Get the upper triangle of the correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find the top correlations
            highest = []
            for col in upper.columns:
                for idx, value in upper[col].items():
                    if not pd.isna(value):
                        highest.append((col, idx, value))
            
            # Sort by absolute correlation value
            highest.sort(key=lambda x: abs(x[2]), reverse=True)
            
            # Add to result
            result["highest_correlations"] = [
                {"column1": col1, "column2": col2, "correlation": corr}
                for col1, col2, corr in highest[:10]  # Top 10
            ]
        
        return result
    
    def _analyze_regression(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """Perform regression analysis on a DataFrame."""
        params = params or {}
        target_column = params.get("target_column")
        feature_columns = params.get("feature_columns")
        
        if not target_column:
            return {"error": "Target column must be specified for regression analysis."}
        
        # Get numeric columns if feature columns not specified
        if not feature_columns:
            numeric_df = df.select_dtypes(include=[np.number])
            feature_columns = [col for col in numeric_df.columns if col != target_column]
        
        # Check if target column exists
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in data."}
        
        # Check if feature columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"Feature columns {missing_columns} not found in data."}
        
        try:
            # Import statsmodels for regression analysis
            import statsmodels.api as sm
            
            # Prepare data
            X = df[feature_columns]
            y = df[target_column]
            
            # Add constant for intercept
            X = sm.add_constant(X)
            
            # Fit the model
            model = sm.OLS(y, X).fit()
            
            # Get results
            result = {
                "summary": str(model.summary()),
                "params": model.params.to_dict(),
                "rsquared": model.rsquared,
                "rsquared_adj": model.rsquared_adj,
                "pvalues": model.pvalues.to_dict(),
                "feature_importance": {}
            }
            
            # Calculate feature importance (normalized absolute coefficients)
            coef_abs = np.abs(model.params[1:])  # Skip intercept
            coef_abs_sum = coef_abs.sum()
            if coef_abs_sum > 0:
                for i, col in enumerate(feature_columns):
                    result["feature_importance"][col] = coef_abs[i] / coef_abs_sum
            
            return result
        
        except Exception as e:
            logger.error(f"Regression analysis failed: {str(e)}")
            return {"error": f"Regression analysis failed: {str(e)}"}
    
    def _analyze_time_series(self, df: pd.DataFrame, params: Optional[Dict] = None) -> Dict:
        """Perform time series analysis on a DataFrame."""
        params = params or {}
        date_column = params.get("date_column")
        value_column = params.get("value_column")
        
        if not date_column:
            # Try to find a date column
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_column = col
                    break
            
            if not date_column:
                return {"error": "Date column must be specified for time series analysis."}
        
        if not value_column:
            # Use the first numeric column that's not the date column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != date_column:
                    value_column = col
                    break
            
            if not value_column:
                return {"error": "Value column must be specified for time series analysis."}
        
        # Check if columns exist
        if date_column not in df.columns:
            return {"error": f"Date column '{date_column}' not found in data."}
        
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found in data."}
        
        try:
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df[date_column] = pd.to_datetime(df[date_column])
            
            # Sort by date
            df = df.sort_values(date_column)
            
            # Set date as index
            ts_df = df.set_index(date_column)[[value_column]]
            
            # Calculate basic statistics
            result = {
                "count": len(ts_df),
                "start_date": ts_df.index.min().isoformat(),
                "end_date": ts_df.index.max().isoformat(),
                "duration": str(ts_df.index.max() - ts_df.index.min()),
                "min_value": float(ts_df[value_column].min()),
                "max_value": float(ts_df[value_column].max()),
                "mean_value": float(ts_df[value_column].mean()),
                "std_value": float(ts_df[value_column].std())
            }
            
            # Calculate trend
            try:
                from scipy import stats
                x = np.arange(len(ts_df))
                y = ts_df[value_column].values
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                result["trend"] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "std_err": std_err,
                    "direction": "increasing" if slope > 0 else "decreasing"
                }
            except Exception as e:
                logger.warning(f"Could not calculate trend: {str(e)}")
            
            # Calculate seasonality if enough data points
            if len(ts_df) >= 4:  # Need at least a few data points
                try:
                    # Resample to regular intervals if needed
                    if not ts_df.index.is_regular:
                        # Determine frequency
                        freq = pd.infer_freq(ts_df.index)
                        if freq:
                            ts_df = ts_df.resample(freq).mean()
                    
                    # Calculate autocorrelation
                    from pandas.plotting import autocorrelation_plot
                    import io
                    import matplotlib.pyplot as plt
                    
                    # Calculate autocorrelation values
                    acf_values = ts_df[value_column].autocorr(lag=1)
                    
                    result["seasonality"] = {
                        "autocorrelation_lag1": float(acf_values)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate seasonality: {str(e)}")
            
            return result
        
        except Exception as e:
            logger.error(f"Time series analysis failed: {str(e)}")
            return {"error": f"Time series analysis failed: {str(e)}"}