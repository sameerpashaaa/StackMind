#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Input Validation Utilities

This module provides validation functions for various types of inputs
to ensure they meet the required criteria before processing.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_text_input(text: str) -> bool:
    """
    Validate text input to ensure it's not empty and within reasonable size limits.
    
    Args:
        text (str): Text input to validate
        
    Returns:
        bool: True if the text input is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        logger.warning("Empty or non-string text input provided")
        return False
    
    # Check if text is within reasonable size limits (e.g., 100KB)
    if len(text) > 100 * 1024:
        logger.warning(f"Text input exceeds size limit: {len(text)} bytes")
        return False
    
    return True

def validate_image_file(file_path: str, allowed_formats: List[str] = None) -> bool:
    """
    Validate an image file to ensure it exists, is accessible, and has an allowed format.
    
    Args:
        file_path (str): Path to the image file
        allowed_formats (List[str], optional): List of allowed file extensions
        
    Returns:
        bool: True if the image file is valid, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.warning(f"Image file does not exist: {file_path}")
        return False
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if file_ext not in allowed_formats:
        logger.warning(f"Unsupported image format: {file_ext}")
        return False
    
    # Check file size (e.g., 10MB limit)
    file_size = os.path.getsize(file_path)
    if file_size > 10 * 1024 * 1024:
        logger.warning(f"Image file exceeds size limit: {file_size} bytes")
        return False
    
    return True

def validate_audio_file(file_path: str, allowed_formats: List[str] = None) -> bool:
    """
    Validate an audio file to ensure it exists, is accessible, and has an allowed format.
    
    Args:
        file_path (str): Path to the audio file
        allowed_formats (List[str], optional): List of allowed file extensions
        
    Returns:
        bool: True if the audio file is valid, False otherwise
    """
    if allowed_formats is None:
        allowed_formats = ['mp3', 'wav', 'ogg', 'm4a', 'flac']
    
    # Check if file exists
    if not os.path.isfile(file_path):
        logger.warning(f"Audio file does not exist: {file_path}")
        return False
    
    # Check file extension
    file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if file_ext not in allowed_formats:
        logger.warning(f"Unsupported audio format: {file_ext}")
        return False
    
    # Check file size (e.g., 50MB limit)
    file_size = os.path.getsize(file_path)
    if file_size > 50 * 1024 * 1024:
        logger.warning(f"Audio file exceeds size limit: {file_size} bytes")
        return False
    
    return True

def validate_code_input(code: str, language: str = None) -> bool:
    """
    Validate code input to ensure it's not empty and within reasonable size limits.
    
    Args:
        code (str): Code input to validate
        language (str, optional): Programming language of the code
        
    Returns:
        bool: True if the code input is valid, False otherwise
    """
    if not code or not isinstance(code, str):
        logger.warning("Empty or non-string code input provided")
        return False
    
    # Check if code is within reasonable size limits (e.g., 500KB)
    if len(code) > 500 * 1024:
        logger.warning(f"Code input exceeds size limit: {len(code)} bytes")
        return False
    
    return True

def validate_url(url: str) -> bool:
    """
    Validate a URL to ensure it has a valid format.
    
    Args:
        url (str): URL to validate
        
    Returns:
        bool: True if the URL is valid, False otherwise
    """
    if not url or not isinstance(url, str):
        logger.warning("Empty or non-string URL provided")
        return False
    
    # Simple URL validation regex
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http://, https://, ftp://, ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # or ipv4
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        logger.warning(f"Invalid URL format: {url}")
        return False
    
    return True

def validate_file_path(file_path: str, must_exist: bool = True) -> bool:
    """
    Validate a file path to ensure it has a valid format and optionally exists.
    
    Args:
        file_path (str): File path to validate
        must_exist (bool): Whether the file must exist
        
    Returns:
        bool: True if the file path is valid, False otherwise
    """
    if not file_path or not isinstance(file_path, str):
        logger.warning("Empty or non-string file path provided")
        return False
    
    # Check if the path is valid
    try:
        path = Path(file_path)
        if must_exist and not path.is_file():
            logger.warning(f"File does not exist: {file_path}")
            return False
    except Exception as e:
        logger.warning(f"Invalid file path: {file_path}, error: {e}")
        return False
    
    return True