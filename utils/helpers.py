#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper Utility Functions

This module provides various utility functions used throughout the AI Problem Solver application.
"""

import os
import re
import json
import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

def get_current_datetime() -> str:
    """
    Get the current date and time as a formatted string.
    
    Returns:
        str: Formatted date and time string
    """
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def generate_unique_id() -> str:
    """
    Generate a unique identifier.
    
    Returns:
        str: Unique identifier
    """
    return str(uuid.uuid4())

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely load a JSON string, returning a default value if parsing fails.
    
    Args:
        json_str (str): JSON string to parse
        default (Any): Default value to return if parsing fails
        
    Returns:
        Any: Parsed JSON object or default value
    """
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely convert an object to a JSON string, returning a default string if conversion fails.
    
    Args:
        obj (Any): Object to convert to JSON
        default (str): Default string to return if conversion fails
        
    Returns:
        str: JSON string representation of the object or default string
    """
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, OverflowError) as e:
        logger.warning(f"Failed to convert to JSON: {e}")
        return default

def ensure_directory_exists(directory_path: str) -> bool:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path (str): Path to the directory
        
    Returns:
        bool: True if the directory exists or was created successfully, False otherwise
    """
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return False

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length, adding a suffix if truncated.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length of the truncated text
        suffix (str): Suffix to add if the text is truncated
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.
    
    Args:
        text (str): Markdown text containing code blocks
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with 'language' and 'code' keys
    """
    pattern = r"```(\w*)\n([\s\S]*?)\n```"
    matches = re.findall(pattern, text)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            "language": language.strip() or "text",
            "code": code.strip()
        })
    
    return code_blocks

def format_time_delta(seconds: float) -> str:
    """
    Format a time delta in seconds to a human-readable string.
    
    Args:
        seconds (float): Time delta in seconds
        
    Returns:
        str: Formatted time string
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"