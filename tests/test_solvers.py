#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Domain Solver Tests

This module contains tests for the domain-specific solvers of the AI Problem Solver.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the application modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from domains.general_solver import GeneralSolver
from domains.math_solver import MathSolver
from domains.code_solver import CodeSolver
from domains.science_solver import ScienceSolver

class TestGeneralSolver(unittest.TestCase):
    """Test case for the general solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = MagicMock(generations=[[MagicMock(text="Test solution")]])
        
        self.general_solver = GeneralSolver(self.mock_llm)
    
    def test_solve(self):
        """Test that the general solver solves problems correctly."""
        problem = "What is the capital of France?"
        result = self.general_solver.solve(problem)
        
        self.assertIsNotNone(result)
        self.assertIn("solution", result)
        self.assertEqual(result["solution"], "Test solution")
    
    def test_create_prompt(self):
        """Test that the general solver creates prompts correctly."""
        problem = "What is the capital of France?"
        prompt = self.general_solver.create_prompt(problem)
        
        self.assertIsNotNone(prompt)
        self.assertIn(problem, prompt)

class TestMathSolver(unittest.TestCase):
    """Test case for the math solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = MagicMock(generations=[[MagicMock(text="x = 2")]])
        
        self.math_solver = MathSolver(self.mock_llm)
    
    def test_solve(self):
        """Test that the math solver solves problems correctly."""
        problem = "Solve for x: 2x + 3 = 7"
        result = self.math_solver.solve(problem)
        
        self.assertIsNotNone(result)
        self.assertIn("solution", result)
        self.assertEqual(result["solution"], "x = 2")
    
    def test_extract_equations(self):
        """Test that the math solver extracts equations correctly."""
        problem = "Solve the equation 2x + 3 = 7 and find the value of y in 3y - 1 = 8"
        equations = self.math_solver.extract_equations(problem)
        
        self.assertIsNotNone(equations)
        self.assertGreater(len(equations), 0)
        self.assertIn("2x + 3 = 7", equations)
        self.assertIn("3y - 1 = 8", equations)
    
    @patch('domains.math_solver.sympy.solve')
    def test_verify_solution(self, mock_solve):
        """Test that the math solver verifies solutions correctly."""
        mock_solve.return_value = [2]
        
        equation = "2*x + 3 - 7"
        solution = "x = 2"
        result = self.math_solver.verify_solution(equation, solution)
        
        self.assertTrue(result)
        mock_solve.assert_called_once()

class TestCodeSolver(unittest.TestCase):
    """Test case for the code solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = MagicMock(generations=[[MagicMock(text="def sum(a, b):\n    return a + b")]])
        
        self.code_solver = CodeSolver(self.mock_llm)
    
    def test_solve(self):
        """Test that the code solver solves problems correctly."""
        problem = "Write a Python function to add two numbers"
        result = self.code_solver.solve(problem)
        
        self.assertIsNotNone(result)
        self.assertIn("solution", result)
        self.assertIn("def sum", result["solution"])
    
    def test_detect_language_from_problem(self):
        """Test that the code solver detects language from problem correctly."""
        problem = "Write a Python function to add two numbers"
        language = self.code_solver.detect_language_from_problem(problem)
        
        self.assertEqual(language, "python")
        
        problem = "Create a JavaScript function to calculate the factorial of a number"
        language = self.code_solver.detect_language_from_problem(problem)
        
        self.assertEqual(language, "javascript")

class TestScienceSolver(unittest.TestCase):
    """Test case for the science solver."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock language model
        self.mock_llm = MagicMock()
        self.mock_llm.generate.return_value = MagicMock(generations=[[MagicMock(text="The acceleration is 9.8 m/s²")]])
        
        self.science_solver = ScienceSolver(self.mock_llm)
    
    def test_solve(self):
        """Test that the science solver solves problems correctly."""
        problem = "Calculate the acceleration of a falling object on Earth"
        result = self.science_solver.solve(problem)
        
        self.assertIsNotNone(result)
        self.assertIn("solution", result)
        self.assertIn("9.8 m/s²", result["solution"])
    
    def test_detect_domain(self):
        """Test that the science solver detects domain correctly."""
        problem = "Calculate the acceleration of a falling object on Earth"
        domain = self.science_solver.detect_domain(problem)
        
        self.assertEqual(domain, "physics")
        
        problem = "Balance the chemical equation: H2 + O2 -> H2O"
        domain = self.science_solver.detect_domain(problem)
        
        self.assertEqual(domain, "chemistry")
        
        problem = "Explain the process of photosynthesis"
        domain = self.science_solver.detect_domain(problem)
        
        self.assertEqual(domain, "biology")

if __name__ == '__main__':
    unittest.main()