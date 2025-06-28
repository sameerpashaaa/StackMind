#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core Functionality Tests

This module contains tests for the core functionality of the AI Problem Solver.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import Settings
from core.memory import MemorySystem
from core.reasoning import ReasoningEngine
from core.planning import PlanningSystem

class TestCoreComponents(unittest.TestCase):
    """Test case for core components of the AI Problem Solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock settings object
        self.settings = MagicMock()
        self.settings.get_section.return_value = {}
        
        # Create a mock language model
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = MagicMock(generations=[[MagicMock(text="Test response")]])
    
    @patch('core.memory.ChromaDB')
    @patch('core.memory.OpenAIEmbeddings')
    def test_memory_system_initialization(self, mock_embeddings, mock_chromadb):
        """Test that the memory system initializes correctly."""
        memory = MemorySystem(self.settings)
        self.assertIsNotNone(memory)
    
    def test_reasoning_engine_initialization(self):
        """Test that the reasoning engine initializes correctly."""
        reasoning = ReasoningEngine(self.mock_llm, self.settings)
        self.assertIsNotNone(reasoning)
    
    def test_planning_system_initialization(self):
        """Test that the planning system initializes correctly."""
        planning = PlanningSystem(self.mock_llm, self.settings)
        self.assertIsNotNone(planning)
    
    @patch('core.reasoning.ReasoningEngine.generate_reasoning')
    def test_reasoning_generation(self, mock_generate_reasoning):
        """Test that the reasoning engine generates reasoning correctly."""
        mock_generate_reasoning.return_value = "Step 1: Understand the problem\nStep 2: Develop a plan\nStep 3: Execute the plan\nStep 4: Verify the solution"
        
        reasoning = ReasoningEngine(self.mock_llm, self.settings)
        result = reasoning.generate_reasoning("What is 2+2?")
        
        self.assertIsNotNone(result)
        self.assertIn("Step 1", result)
        mock_generate_reasoning.assert_called_once()
    
    @patch('core.planning.PlanningSystem.create_plan')
    def test_plan_creation(self, mock_create_plan):
        """Test that the planning system creates plans correctly."""
        mock_plan = {
            "steps": [
                {"id": 1, "description": "Understand the problem"},
                {"id": 2, "description": "Develop a solution approach"},
                {"id": 3, "description": "Implement the solution"},
                {"id": 4, "description": "Verify the results"}
            ]
        }
        mock_create_plan.return_value = mock_plan
        
        planning = PlanningSystem(self.mock_llm, self.settings)
        result = planning.create_plan("Solve the equation 2x + 3 = 7")
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result["steps"]), 4)
        mock_create_plan.assert_called_once()

if __name__ == '__main__':
    unittest.main()