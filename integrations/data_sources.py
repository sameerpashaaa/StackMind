#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
External Data Sources Integration Module

This module provides integration with external data sources such as APIs,
databases, and web services that can be used by the AI Problem Solver.
"""

import os
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union

import requests
from bs4 import BeautifulSoup
import pandas as pd
import feedparser
import yfinance as yf
from newsapi import NewsApiClient

from utils.helpers import safe_json_loads, safe_json_dumps
from utils.validators import validate_url

logger = logging.getLogger(__name__)

# Cache for storing API responses to reduce redundant requests
API_CACHE = {}
CACHE_EXPIRY = 300  # Cache expiry time in seconds (5 minutes)

# Rate limiting settings
RATE_LIMITS = {
    "default": {"requests": 10, "period": 60},  # 10 requests per minute
    "news_api": {"requests": 5, "period": 60},  # 5 requests per minute
    "financial_api": {"requests": 5, "period": 60}  # 5 requests per minute
}

# Request timestamps for rate limiting
REQUEST_TIMESTAMPS = {}

def _check_rate_limit(source: str) -> bool:
    """
    Check if a request to a specific source would exceed the rate limit.
    
    Args:
        source (str): Name of the data source
        
    Returns:
        bool: True if the request is allowed, False if it would exceed the rate limit
    """
    now = time.time()
    source_limits = RATE_LIMITS.get(source, RATE_LIMITS["default"])
    
    if source not in REQUEST_TIMESTAMPS:
        REQUEST_TIMESTAMPS[source] = []
    
    # Remove timestamps older than the rate limit period
    REQUEST_TIMESTAMPS[source] = [ts for ts in REQUEST_TIMESTAMPS[source] if now - ts <= source_limits["period"]]
    
    # Check if adding a new request would exceed the rate limit
    if len(REQUEST_TIMESTAMPS[source]) >= source_limits["requests"]:
        logger.warning(f"Rate limit exceeded for {source}")
        return False
    
    # Add the current timestamp
    REQUEST_TIMESTAMPS[source].append(now)
    return True

def _get_cached_response(cache_key: str) -> Optional[Dict]:
    """
    Get a cached response if it exists and hasn't expired.
    
    Args:
        cache_key (str): Cache key
        
    Returns:
        Optional[Dict]: Cached response or None if not found or expired
    """
    if cache_key in API_CACHE:
        cached_data = API_CACHE[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_EXPIRY:
            return cached_data["data"]
    
    return None

def _cache_response(cache_key: str, data: Any) -> None:
    """
    Cache a response with the current timestamp.
    
    Args:
        cache_key (str): Cache key
        data (Any): Data to cache
    """
    API_CACHE[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }

def fetch_rest_api(url: str, method: str = "GET", params: Dict = None, headers: Dict = None, cache: bool = True) -> Dict:
    """
    Fetch data from a REST API.
    
    Args:
        url (str): API URL
        method (str): HTTP method (GET, POST, etc.)
        params (Dict): Query parameters
        headers (Dict): HTTP headers
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: API response data
    """
    try:
        if not validate_url(url):
            return {"error": "Invalid URL"}
        
        # Check rate limit
        if not _check_rate_limit("default"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"{method}:{url}:{json.dumps(params) if params else ''}:{json.dumps(headers) if headers else ''}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached response for {url}")
                return cached_response
        
        # Make the request
        response = requests.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        
        # Parse the response
        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
        else:
            data = {"text": response.text}
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from {url}: {e}")
        return {"error": str(e)}

def fetch_graphql_api(url: str, query: str, variables: Dict = None, headers: Dict = None, cache: bool = True) -> Dict:
    """
    Fetch data from a GraphQL API.
    
    Args:
        url (str): API URL
        query (str): GraphQL query
        variables (Dict): Query variables
        headers (Dict): HTTP headers
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: API response data
    """
    try:
        if not validate_url(url):
            return {"error": "Invalid URL"}
        
        # Check rate limit
        if not _check_rate_limit("default"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"graphql:{url}:{query}:{json.dumps(variables) if variables else ''}:{json.dumps(headers) if headers else ''}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached response for GraphQL query to {url}")
                return cached_response
        
        # Prepare the request
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        # Make the request
        response = requests.post(
            url=url,
            json=payload,
            headers=headers,
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from GraphQL API {url}: {e}")
        return {"error": str(e)}

def fetch_web_page(url: str, cache: bool = True) -> Dict:
    """
    Fetch and parse a web page using BeautifulSoup.
    
    Args:
        url (str): Web page URL
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: Parsed web page data
    """
    try:
        if not validate_url(url):
            return {"error": "Invalid URL"}
        
        # Check rate limit
        if not _check_rate_limit("default"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"web:{url}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached response for web page {url}")
                return cached_response
        
        # Make the request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract useful information
        data = {
            "title": soup.title.string if soup.title else "",
            "text": soup.get_text(separator="\n", strip=True),
            "links": [a.get("href") for a in soup.find_all("a") if a.get("href")],
            "html": response.text
        }
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch web page {url}: {e}")
        return {"error": str(e)}

def fetch_financial_data(symbol: str, period: str = "1y", interval: str = "1d", cache: bool = True) -> Dict:
    """
    Fetch financial data for a stock symbol using yfinance.
    
    Args:
        symbol (str): Stock symbol
        period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: Financial data
    """
    try:
        # Check rate limit
        if not _check_rate_limit("financial_api"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"financial:{symbol}:{period}:{interval}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached financial data for {symbol}")
                return cached_response
        
        # Fetch the data
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval=interval)
        
        # Convert to dictionary
        data = {
            "symbol": symbol,
            "info": ticker.info,
            "history": json.loads(history.to_json(orient="table"))
        }
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to fetch financial data for {symbol}: {e}")
        return {"error": str(e)}

def fetch_news(query: str = None, sources: str = None, language: str = "en", page_size: int = 10, cache: bool = True) -> Dict:
    """
    Fetch news articles using the News API.
    
    Args:
        query (str): Search query
        sources (str): Comma-separated list of news sources
        language (str): Language code
        page_size (int): Number of articles to fetch
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: News articles
    """
    try:
        # Check for API key
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            logger.warning("NEWS_API_KEY not found in environment variables")
            return {"error": "NEWS_API_KEY not found in environment variables"}
        
        # Check rate limit
        if not _check_rate_limit("news_api"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"news:{query}:{sources}:{language}:{page_size}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached news data for query '{query}'")
                return cached_response
        
        # Initialize the client
        newsapi = NewsApiClient(api_key=api_key)
        
        # Fetch the data
        if query:
            data = newsapi.get_everything(
                q=query,
                sources=sources,
                language=language,
                page_size=page_size
            )
        else:
            data = newsapi.get_top_headlines(
                sources=sources,
                language=language,
                page_size=page_size
            )
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        return {"error": str(e)}

def fetch_rss_feed(url: str, cache: bool = True) -> Dict:
    """
    Fetch and parse an RSS feed.
    
    Args:
        url (str): RSS feed URL
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: Parsed RSS feed data
    """
    try:
        if not validate_url(url):
            return {"error": "Invalid URL"}
        
        # Check rate limit
        if not _check_rate_limit("default"):
            return {"error": "Rate limit exceeded"}
        
        # Check cache
        cache_key = f"rss:{url}"
        if cache:
            cached_response = _get_cached_response(cache_key)
            if cached_response:
                logger.info(f"Using cached RSS feed data for {url}")
                return cached_response
        
        # Parse the feed
        feed = feedparser.parse(url)
        
        # Extract useful information
        data = {
            "title": feed.feed.get("title", ""),
            "link": feed.feed.get("link", ""),
            "description": feed.feed.get("description", ""),
            "entries": [
                {
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", ""),
                    "author": entry.get("author", "")
                }
                for entry in feed.entries
            ]
        }
        
        # Cache the response
        if cache:
            _cache_response(cache_key, data)
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to fetch RSS feed {url}: {e}")
        return {"error": str(e)}

def fetch_weather(location: str, cache: bool = True) -> Dict:
    """
    Fetch weather data for a location using a weather API.
    
    Args:
        location (str): Location name or coordinates
        cache (bool): Whether to cache the response
        
    Returns:
        Dict: Weather data
    """
    try:
        # This is a placeholder for a weather API integration
        # You would need to sign up for a weather API service and use their API
        
        # For now, we'll return a mock response
        logger.warning("Weather API not implemented, returning mock data")
        
        return {
            "location": location,
            "temperature": 22.5,
            "humidity": 65,
            "conditions": "Partly Cloudy",
            "forecast": [
                {"day": "Today", "high": 24, "low": 18, "conditions": "Partly Cloudy"},
                {"day": "Tomorrow", "high": 26, "low": 19, "conditions": "Sunny"}
            ],
            "note": "This is mock data as the weather API is not implemented"
        }
    
    except Exception as e:
        logger.error(f"Failed to fetch weather data for {location}: {e}")
        return {"error": str(e)}