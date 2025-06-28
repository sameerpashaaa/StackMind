#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Tests

This module contains tests for the interfaces of the AI Problem Solver.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interfaces.cli import CommandLineInterface
from interfaces.api import app, start_api_server

class TestCommandLineInterface(unittest.TestCase):
    """Test case for the command-line interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock agent
        self.mock_agent = MagicMock()
        self.mock_agent.solve_problem.return_value = {
            "solution": "Test solution",
            "reasoning": "Test reasoning",
            "steps": ["Step 1", "Step 2"],
            "domain": "general"
        }
        
        # Create a mock settings object
        self.mock_settings = MagicMock()
        self.mock_settings.get_section.return_value = {}
        
        self.cli = CommandLineInterface(self.mock_agent, self.mock_settings)
    
    @patch('interfaces.cli.input')
    @patch('interfaces.cli.print')
    def test_process_text_input(self, mock_print, mock_input):
        """Test that the CLI processes text input correctly."""
        mock_input.return_value = "What is 2+2?"
        
        self.cli.process_text_input()
        
        self.mock_agent.solve_problem.assert_called_once()
        mock_print.assert_called()
    
    @patch('interfaces.cli.input')
    @patch('interfaces.cli.print')
    def test_display_solution(self, mock_print, mock_input):
        """Test that the CLI displays solutions correctly."""
        solution = {
            "solution": "The answer is 4",
            "reasoning": "2 + 2 = 4",
            "steps": ["Step 1: Add 2 and 2", "Step 2: Get 4"],
            "domain": "math"
        }
        
        self.cli.display_solution(solution)
        
        # Check that print was called multiple times
        self.assertGreater(mock_print.call_count, 3)

class TestAPIInterface(unittest.TestCase):
    """Test case for the API interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock agent
        self.mock_agent = MagicMock()
        self.mock_agent.solve_problem.return_value = {
            "solution": "Test solution",
            "reasoning": "Test reasoning",
            "steps": ["Step 1", "Step 2"],
            "domain": "general"
        }
        
        # Create a mock settings object
        self.mock_settings = MagicMock()
        self.mock_settings.get_section.return_value = {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False
        }
    
    @patch('interfaces.api.app')
    @patch('interfaces.api.uvicorn.run')
    def test_start_api_server(self, mock_run, mock_app):
        """Test that the API server starts correctly."""
        start_api_server(self.mock_agent, self.mock_settings)
        
        mock_run.assert_called_once()
    
    @patch('interfaces.api.app.state.agent')
    def test_solve_problem_endpoint(self, mock_agent):
        """Test the solve problem endpoint."""
        # This is a placeholder for testing FastAPI endpoints
        # In a real test, you would use TestClient from fastapi.testclient
        pass

if __name__ == '__main__':
    unittest.main()