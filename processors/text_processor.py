#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Text Processor Module

This module implements text processing capabilities for the AI Problem Solver,
allowing it to handle and analyze text-based inputs.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Text processor for handling text-based inputs.
    
    This processor analyzes and extracts information from text inputs,
    including natural language, structured text, and code snippets.
    
    Attributes:
        None
    """
    
    def __init__(self):
        """
        Initialize the Text Processor.
        """
        logger.info("Text processor initialized")
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Process text input and extract relevant information.
        
        Args:
            text: The text input to process
            
        Returns:
            Dict[str, Any]: Processed text with metadata
        """
        # Initialize result
        result = {
            "content": text,
            "content_type": "text",
            "metadata": {}
        }
        
        # Extract basic metadata
        result["metadata"]["length"] = len(text)
        result["metadata"]["word_count"] = len(text.split())
        
        # Detect language features
        result["metadata"]["has_code"] = self._detect_code(text)
        result["metadata"]["has_math"] = self._detect_math(text)
        result["metadata"]["has_urls"] = self._detect_urls(text)
        
        # Detect text structure
        result["metadata"]["structure"] = self._detect_structure(text)
        
        # Extract entities if present
        result["metadata"]["entities"] = self._extract_entities(text)
        
        logger.debug(f"Processed text input: {len(text)} chars, {result['metadata']['word_count']} words")
        return result
    
    def _detect_code(self, text: str) -> bool:
        """
        Detect if the text contains code snippets.
        
        Args:
            text: The text to analyze
            
        Returns:
            bool: Whether code is detected
        """
        # Check for code block markers
        if re.search(r'```[\w]*\n[\s\S]*?```', text):
            return True
        
        # Check for common code patterns
        code_patterns = [
            r'\bfunction\s+\w+\s*\(',  # function declarations
            r'\bdef\s+\w+\s*\(',      # Python function declarations
            r'\bclass\s+\w+',         # class declarations
            r'\bif\s*\(.+\)\s*\{',   # if statements with braces
            r'\bfor\s*\(.+\)\s*\{',  # for loops with braces
            r'\bwhile\s*\(.+\)\s*\{', # while loops with braces
            r'\bimport\s+[\w\.]+;',   # import statements
            r'\bfrom\s+[\w\.]+\s+import', # Python imports
            r'<[\w\s="\'\-\:\/\.]+>.*?</[\w]+>', # HTML tags
            r'\$\(.+\)\.',           # jQuery
            r'\bvar\s+\w+\s*=',      # JavaScript variable declarations
            r'\blet\s+\w+\s*=',      # JavaScript let declarations
            r'\bconst\s+\w+\s*='     # JavaScript const declarations
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_math(self, text: str) -> bool:
        """
        Detect if the text contains mathematical expressions.
        
        Args:
            text: The text to analyze
            
        Returns:
            bool: Whether math is detected
        """
        # Check for LaTeX-style math delimiters
        if re.search(r'\$\$[\s\S]*?\$\$', text) or re.search(r'\$[^\$\n]+\$', text):
            return True
        
        # Check for common math patterns
        math_patterns = [
            r'\b\d+\s*[+\-*/^]\s*\d+',  # Basic arithmetic
            r'\b\d+\s*=\s*\d+',         # Equations
            r'\b\d+\s*>\s*\d+',         # Inequalities
            r'\b\d+\s*<\s*\d+',         # Inequalities
            r'\b\d+\s*≤\s*\d+',         # Inequalities with symbols
            r'\b\d+\s*≥\s*\d+',         # Inequalities with symbols
            r'\b\d+\s*±\s*\d+',         # Plus-minus
            r'\b\d+\s*×\s*\d+',         # Multiplication symbol
            r'\b\d+\s*÷\s*\d+',         # Division symbol
            r'\b\d+\s*\^\s*\d+',        # Exponentiation
            r'\bsqrt\s*\(',              # Square root
            r'\bsin\s*\(',               # Trigonometric functions
            r'\bcos\s*\(',
            r'\btan\s*\(',
            r'\blog\s*\(',               # Logarithm
            r'\bln\s*\(',                # Natural logarithm
            r'\bsum\s*\(',               # Summation
            r'\bint\s*\(',               # Integral
            r'\blim\s*\(',               # Limit
            r'\b\d+\s*!',                # Factorial
            r'\b\(\s*\d+\s*,\s*\d+\s*\)', # Coordinates
            r'\b\[\s*\d+\s*,\s*\d+\s*\]', # Vectors
            r'\b\{\s*\d+\s*,\s*\d+\s*\}', # Sets
            r'\b\d+\s*\\times\s*\d+'    # LaTeX multiplication
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _detect_urls(self, text: str) -> bool:
        """
        Detect if the text contains URLs.
        
        Args:
            text: The text to analyze
            
        Returns:
            bool: Whether URLs are detected
        """
        # URL pattern
        url_pattern = r'https?://[\w\d\-\.]+\.[a-zA-Z]{2,}[\w\d\-\.\/\?\=\&\%\+\#\~\;\:\@\,]*'
        
        return bool(re.search(url_pattern, text))
    
    def _detect_structure(self, text: str) -> str:
        """
        Detect the structure of the text.
        
        Args:
            text: The text to analyze
            
        Returns:
            str: Detected structure type
        """
        # Check for markdown headings
        if re.search(r'^\s*#\s+.+$', text, re.MULTILINE):
            return "markdown"
        
        # Check for bullet points
        if re.search(r'^\s*[\*\-\+]\s+.+$', text, re.MULTILINE):
            return "bullet_list"
        
        # Check for numbered lists
        if re.search(r'^\s*\d+\.\s+.+$', text, re.MULTILINE):
            return "numbered_list"
        
        # Check for Q&A format
        if re.search(r'^\s*Q\s*:\s*.+$', text, re.MULTILINE) and re.search(r'^\s*A\s*:\s*.+$', text, re.MULTILINE):
            return "qa_format"
        
        # Check for JSON-like structure
        if text.strip().startswith('{') and text.strip().endswith('}') and ':' in text:
            return "json_like"
        
        # Check for table-like structure
        if re.search(r'^\s*\|.+\|\s*$', text, re.MULTILINE) and re.search(r'^\s*\|\s*[-:]+\s*\|\s*[-:]+', text, re.MULTILINE):
            return "table"
        
        # Default to paragraph
        return "paragraph"
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from the text.
        
        This is a simple implementation that could be enhanced with NLP libraries.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict[str, List[str]]: Extracted entities by type
        """
        entities = {
            "emails": [],
            "urls": [],
            "dates": [],
            "numbers": []
        }
        
        # Extract emails
        email_pattern = r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}'
        entities["emails"] = re.findall(email_pattern, text)
        
        # Extract URLs
        url_pattern = r'https?://[\w\d\-\.]+\.[a-zA-Z]{2,}[\w\d\-\.\/\?\=\&\%\+\#\~\;\:\@\,]*'
        entities["urls"] = re.findall(url_pattern, text)
        
        # Extract dates (simple patterns)
        date_patterns = [
            r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
            r'\d{4}[/\-]\d{1,2}[/\-]\d{1,2}',  # YYYY/MM/DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}\s*,?\s*\d{2,4}\b',  # Month DD, YYYY
            r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b'  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            entities["dates"].extend(re.findall(pattern, text, re.IGNORECASE))
        
        # Extract numbers
        # This will find integers, decimals, and numbers with commas as thousand separators
        number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, text)
        
        return entities
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Extract keywords from the text.
        
        This is a simple implementation that could be enhanced with NLP libraries.
        
        Args:
            text: The text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List[str]: Extracted keywords
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b[\w\-\']+\b', text.lower())
        
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
            'when', 'where', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
            'then', 'just', 'so', 'than', 'such', 'both', 'through', 'about', 'for',
            'is', 'of', 'while', 'during', 'to', 'from', 'in', 'on', 'at', 'by', 'with',
            'be', 'been', 'being', 'am', 'are', 'was', 'were', 'has', 'have', 'had',
            'do', 'does', 'did', 'can', 'could', 'will', 'would', 'should', 'shall',
            'may', 'might', 'must', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = {}
        for word in filtered_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, count in sorted_words[:max_keywords]]
        
        return keywords
    
    def summarize(self, text: str, max_length: int = 200) -> str:
        """
        Generate a simple summary of the text.
        
        This is a basic extractive summarization that could be enhanced with NLP libraries.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in characters
            
        Returns:
            str: Text summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return ""
        
        # If text is already short, return it
        if len(text) <= max_length:
            return text
        
        # Simple scoring: prefer sentences at the beginning and end
        # and sentences containing keywords
        keywords = self.extract_keywords(text, max_keywords=20)
        
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            # Base score: position score (beginning and end get higher scores)
            position_score = 1.0
            if i < 3:  # First three sentences
                position_score = 1.5 - (i * 0.2)
            elif i >= len(sentences) - 3:  # Last three sentences
                position_score = 1.0 + ((i - (len(sentences) - 3)) * 0.1)
            
            # Keyword score: sentences with more keywords get higher scores
            keyword_score = 0
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    keyword_score += 1
            
            # Length score: prefer medium-length sentences
            length = len(sentence)
            length_score = 1.0
            if length < 10:  # Too short
                length_score = 0.5
            elif length > 100:  # Too long
                length_score = 0.7
            
            # Calculate total score
            total_score = position_score + (keyword_score * 0.2) + length_score
            
            scored_sentences.append((sentence, total_score))
        
        # Sort sentences by score (highest first)
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
        
        # Build summary by adding sentences until we reach max_length
        summary = ""
        for sentence, _ in sorted_sentences:
            if len(summary) + len(sentence) + 1 <= max_length:
                summary += sentence + " "
            else:
                break
        
        return summary.strip()