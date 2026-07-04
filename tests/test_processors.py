#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processor Tests

This module contains tests for the input processors of the AI Problem Solver.
"""

import os
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from processors.code_processor import CodeProcessor
from processors.image_processor import ImageProcessor
from processors.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Test case for the text processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.text_processor = TextProcessor()

    def test_process_text(self):
        """Test that the text processor processes text correctly."""
        result = self.text_processor.process("This is a test.")
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertEqual(result["content"], "This is a test.")


    def test_extract_entities(self):
        """Test that the text processor extracts entities correctly."""
        text = "Contact john.doe@example.com or visit https://example.com on January 15, 2023."
        result = self.text_processor._extract_entities(text)

        self.assertIn("emails", result)
        self.assertIn("urls", result)
        self.assertIn("dates", result)

        self.assertIn("john.doe@example.com", result["emails"])
        self.assertIn("https://example.com", result["urls"])
        self.assertIn("January 15, 2023", result["dates"])


class TestImageProcessor(unittest.TestCase):
    """Test case for the image processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.image_processor = ImageProcessor()

    @patch("processors.image_processor.Image.open")
    def test_extract_metadata(self, mock_image_open):
        """Test that the image processor extracts metadata correctly."""
        # Mock the image object
        mock_image = MagicMock()
        mock_image.format = "JPEG"
        mock_image.size = (800, 600)
        mock_image.width = 800
        mock_image.height = 600
        mock_image.mode = "RGB"
        mock_image_open.return_value = mock_image

        result = self.image_processor._extract_metadata(mock_image)

        self.assertIn("format", result)
        self.assertIn("width", result)
        self.assertIn("height", result)
        self.assertIn("mode", result)

        self.assertEqual(result["format"], "JPEG")
        self.assertEqual(result["width"], 800)
        self.assertEqual(result["height"], 600)
        self.assertEqual(result["mode"], "RGB")

    @patch("pytesseract.image_to_string")
    @patch("processors.image_processor.Image.open")
    def test_extract_text(self, mock_image_open, mock_image_to_string):
        """Test that the image processor extracts text correctly."""
        # Mock the OCR function
        mock_image_to_string.return_value = "Extracted text from image"

        # Mock the image object
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        result = self.image_processor._extract_text("test.jpg")
        self.assertEqual(result, "Extracted text from image")
        mock_image_to_string.assert_called_once()


class TestCodeProcessor(unittest.TestCase):
    """Test case for the code processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.code_processor = CodeProcessor()

    def test_detect_language(self):
        """Test that the code processor detects language correctly."""
        python_code = "def add(a, b):\n    return a + b"
        result = self.code_processor._detect_language(python_code)
        self.assertEqual(result, "python")

        js_code = "function add(a, b) {\n    return a + b;\n}"
        result = self.code_processor._detect_language(js_code)
        self.assertEqual(result, "javascript")

    def test_analyze_structure(self):
        """Test that the code processor analyzes structure correctly."""
        python_code = """import os
import sys

def hello(name):
    return f"Hello, {name}!"

class Person:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return hello(self.name)

if __name__ == "__main__":
    person = Person("World")
    print(person.greet())
"""
        result = self.code_processor._analyze_structure(python_code, "python")

        self.assertIsNotNone(result)
        self.assertIn("functions", result)
        self.assertIn("classes", result)
        self.assertIn("add" if "add" in result["functions"] else "hello", result["functions"])


if __name__ == "__main__":
    unittest.main()
