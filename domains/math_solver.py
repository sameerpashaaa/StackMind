#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Math Solver Module

This module implements mathematical problem-solving capabilities for the AI Problem Solver,
allowing it to handle various types of math problems, from basic arithmetic to advanced
calculus, algebra, and statistics.
"""

import logging
import re
import sympy
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from sympy import symbols, solve, simplify, expand, factor, integrate, diff, Matrix
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import latex

logger = logging.getLogger(__name__)

class MathSolver:
    """
    Math solver for handling mathematical problems.
    
    This solver can handle various types of mathematical problems, including:
    - Arithmetic calculations
    - Algebraic equations and expressions
    - Calculus (differentiation, integration)
    - Linear algebra (matrices, vectors)
    - Statistics and probability
    - Geometry
    
    Attributes:
        llm: Language model for enhanced problem understanding and explanation
        settings: Application settings
    """
    
    def __init__(self, llm=None, settings=None):
        """
        Initialize the Math Solver.
        
        Args:
            llm: Language model for enhanced problem understanding and explanation
            settings: Application settings
        """
        self.llm = llm
        self.settings = settings
        
        # Initialize problem type patterns
        self.problem_patterns = self._initialize_problem_patterns()
        
        logger.info("Math solver initialized")
    
    def solve(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve a mathematical problem.
        
        Args:
            problem: The mathematical problem to solve
            context: Additional context for the problem
            
        Returns:
            Dict[str, Any]: Solution with steps and explanation
        """
        # Initialize result
        result = {
            "problem": problem,
            "solution": None,
            "steps": [],
            "explanation": "",
            "visualization": None,
            "error": None,
            "problem_type": None
        }
        
        try:
            # Detect problem type
            problem_type = self._detect_problem_type(problem)
            result["problem_type"] = problem_type
            
            # Extract mathematical expressions
            expressions = self._extract_expressions(problem)
            
            # Solve based on problem type
            if problem_type == "arithmetic":
                solution_data = self._solve_arithmetic(problem, expressions)
            elif problem_type == "algebraic_equation":
                solution_data = self._solve_algebraic_equation(problem, expressions)
            elif problem_type == "algebraic_expression":
                solution_data = self._solve_algebraic_expression(problem, expressions)
            elif problem_type == "calculus_differentiation":
                solution_data = self._solve_differentiation(problem, expressions)
            elif problem_type == "calculus_integration":
                solution_data = self._solve_integration(problem, expressions)
            elif problem_type == "linear_algebra":
                solution_data = self._solve_linear_algebra(problem, expressions)
            elif problem_type == "statistics":
                solution_data = self._solve_statistics(problem, expressions)
            elif problem_type == "geometry":
                solution_data = self._solve_geometry(problem, expressions)
            else:
                # Use LLM for general math problems or undetected types
                solution_data = self._solve_with_llm(problem, context)
            
            # Update result with solution data
            result.update(solution_data)
            
            # Generate explanation if not already provided
            if not result["explanation"] and self.llm:
                result["explanation"] = self._generate_explanation(problem, result["solution"], result["steps"])
            
            # Generate visualization if applicable
            if self._can_visualize(problem_type, result["solution"]):
                result["visualization"] = self._generate_visualization(problem_type, result["solution"])
            
        except Exception as e:
            logger.error(f"Error solving math problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def verify_solution(self, problem: str, solution: Any) -> Dict[str, Any]:
        """
        Verify a mathematical solution.
        
        Args:
            problem: The original mathematical problem
            solution: The solution to verify
            
        Returns:
            Dict[str, Any]: Verification results
        """
        result = {
            "verified": False,
            "explanation": "",
            "alternative_solution": None
        }
        
        try:
            # Detect problem type
            problem_type = self._detect_problem_type(problem)
            
            # Extract mathematical expressions
            expressions = self._extract_expressions(problem)
            
            # Verify based on problem type
            if problem_type == "arithmetic":
                # For arithmetic, recalculate and compare
                solution_data = self._solve_arithmetic(problem, expressions)
                result["verified"] = self._compare_solutions(solution, solution_data["solution"])
            elif problem_type == "algebraic_equation":
                # For equations, substitute the solution back into the equation
                result["verified"] = self._verify_equation_solution(expressions, solution)
            elif problem_type == "algebraic_expression":
                # For expressions, compare simplified forms
                solution_data = self._solve_algebraic_expression(problem, expressions)
                result["verified"] = self._compare_expressions(solution, solution_data["solution"])
            else:
                # Use LLM for verification of other problem types
                if self.llm:
                    result = self._verify_with_llm(problem, solution)
                else:
                    result["explanation"] = "Verification not available for this problem type without LLM."
            
            # Generate alternative solution if verification failed
            if not result["verified"] and self.llm:
                alt_solution_data = self.solve(problem)
                if alt_solution_data["solution"] and not alt_solution_data["error"]:
                    result["alternative_solution"] = alt_solution_data
            
        except Exception as e:
            logger.error(f"Error verifying math solution: {str(e)}")
            result["explanation"] = f"Verification error: {str(e)}"
        
        return result
    
    def _initialize_problem_patterns(self) -> Dict[str, List[str]]:
        """
        Initialize patterns for detecting different types of math problems.
        
        Returns:
            Dict[str, List[str]]: Problem type patterns
        """
        return {
            "arithmetic": [
                r'\d+\s*[+\-*/^]\s*\d+',
                r'calculate',
                r'compute',
                r'evaluate',
                r'what is',
                r'find the value',
                r'sum of',
                r'product of',
                r'difference between',
                r'quotient of'
            ],
            "algebraic_equation": [
                r'solve for',
                r'find x',
                r'equation',
                r'=',
                r'unknown',
                r'variable',
                r'solve the equation',
                r'find the value of',
                r'find the root',
                r'quadratic',
                r'linear equation'
            ],
            "algebraic_expression": [
                r'simplify',
                r'expand',
                r'factor',
                r'distribute',
                r'expression',
                r'polynomial',
                r'simplify the expression',
                r'expand the expression',
                r'factor the expression'
            ],
            "calculus_differentiation": [
                r'derivative',
                r'differentiate',
                r'rate of change',
                r'slope',
                r'tangent',
                r'maximum',
                r'minimum',
                r'critical point',
                r'find the derivative',
                r'd/dx',
                r'f\'\(x\)'
            ],
            "calculus_integration": [
                r'integral',
                r'integrate',
                r'antiderivative',
                r'area under',
                r'find the integral',
                r'\\int',
                r'∫'
            ],
            "linear_algebra": [
                r'matrix',
                r'vector',
                r'determinant',
                r'eigenvalue',
                r'eigenvector',
                r'linear system',
                r'system of equations',
                r'inverse matrix',
                r'transpose',
                r'dot product',
                r'cross product'
            ],
            "statistics": [
                r'mean',
                r'median',
                r'mode',
                r'standard deviation',
                r'variance',
                r'probability',
                r'distribution',
                r'normal distribution',
                r'binomial',
                r'poisson',
                r'confidence interval',
                r'hypothesis test',
                r'p-value',
                r'correlation',
                r'regression'
            ],
            "geometry": [
                r'area',
                r'perimeter',
                r'volume',
                r'surface area',
                r'angle',
                r'triangle',
                r'circle',
                r'square',
                r'rectangle',
                r'polygon',
                r'sphere',
                r'cube',
                r'cylinder',
                r'cone',
                r'distance',
                r'pythagorean'
            ]
        }
    
    def _detect_problem_type(self, problem: str) -> str:
        """
        Detect the type of mathematical problem.
        
        Args:
            problem: The mathematical problem
            
        Returns:
            str: Detected problem type
        """
        problem_lower = problem.lower()
        
        # Score each problem type based on pattern matches
        scores = {}
        for problem_type, patterns in self.problem_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, problem_lower, re.IGNORECASE):
                    score += 1
            scores[problem_type] = score
        
        # Get the problem type with the highest score
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                max_types = [pt for pt, score in scores.items() if score == max_score]
                return max_types[0]  # Return the first type if there's a tie
        
        # Default to general if no specific type is detected
        return "general"
    
    def _extract_expressions(self, problem: str) -> List[str]:
        """
        Extract mathematical expressions from the problem text.
        
        Args:
            problem: The mathematical problem
            
        Returns:
            List[str]: Extracted expressions
        """
        expressions = []
        
        # Extract expressions enclosed in $ or $$ (LaTeX style)
        latex_expressions = re.findall(r'\$(.*?)\$', problem)
        expressions.extend(latex_expressions)
        
        # Extract expressions that look like equations (contains =)
        equation_expressions = re.findall(r'([\w\d\s+\-*/^(){}\[\]]+=[\w\d\s+\-*/^(){}\[\]]+)', problem)
        expressions.extend(equation_expressions)
        
        # Extract expressions that look like mathematical formulas
        formula_expressions = re.findall(r'([\d]+[+\-*/^][\d]+|[a-zA-Z]\([a-zA-Z]\)|\\frac\{.*\}\{.*\}|\\sqrt\{.*\})', problem)
        expressions.extend(formula_expressions)
        
        # If no expressions found, use the entire problem text
        if not expressions:
            expressions.append(problem)
        
        return expressions
    
    def _solve_arithmetic(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve arithmetic problems.
        
        Args:
            problem: The arithmetic problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract the arithmetic expression
            expression = expressions[0]
            
            # Clean up the expression
            expression = re.sub(r'[^\d+\-*/^()\s.]', '', expression)
            expression = expression.replace('^', '**')
            
            # Evaluate step by step
            steps = []
            
            # Parse the expression into components
            components = re.findall(r'\d+\.?\d*|[+\-*/^()]', expression)
            current_expr = ''.join(components)
            steps.append(f"Starting with: {current_expr}")
            
            # Handle parentheses first
            while '(' in current_expr:
                # Find innermost parentheses
                inner_expr = re.search(r'\(([^()]+)\)', current_expr)
                if inner_expr:
                    inner_result = eval(inner_expr.group(1))
                    steps.append(f"Evaluate {inner_expr.group(0)} = {inner_result}")
                    current_expr = current_expr.replace(inner_expr.group(0), str(inner_result), 1)
                    steps.append(f"Expression becomes: {current_expr}")
                else:
                    break
            
            # Evaluate the final expression
            final_result = eval(current_expr)
            steps.append(f"Final evaluation: {current_expr} = {final_result}")
            
            result["solution"] = final_result
            result["steps"] = steps
            result["explanation"] = "Arithmetic calculation performed step by step, following order of operations."
            
        except Exception as e:
            logger.error(f"Error solving arithmetic problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_algebraic_equation(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve algebraic equations.
        
        Args:
            problem: The algebraic equation problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract the equation
            equation = expressions[0]
            
            # Clean up the equation
            equation = equation.replace('^', '**')
            
            # Check if it's an equation (contains =)
            if '=' in equation:
                # Split into left and right sides
                left_side, right_side = equation.split('=', 1)
                
                # Convert to SymPy expressions
                left_expr = parse_expr(left_side.strip())
                right_expr = parse_expr(right_side.strip())
                
                # Move everything to the left side
                equation_expr = left_expr - right_expr
                
                # Find all symbols (variables)
                symbols_in_expr = list(equation_expr.free_symbols)
                
                if not symbols_in_expr:
                    result["error"] = "No variables found in the equation."
                    return result
                
                # Solve for the first symbol (usually x)
                symbol_to_solve = symbols_in_expr[0]
                
                # Add steps
                steps = []
                steps.append(f"Original equation: {left_side} = {right_side}")
                steps.append(f"Move all terms to the left side: {left_side} - ({right_side}) = 0")
                steps.append(f"Simplify: {equation_expr} = 0")
                steps.append(f"Solve for {symbol_to_solve}")
                
                # Solve the equation
                solutions = solve(equation_expr, symbol_to_solve)
                
                if solutions:
                    steps.append(f"Solutions: {symbol_to_solve} = {', '.join(str(sol) for sol in solutions)}")
                    result["solution"] = {str(symbol_to_solve): [str(sol) for sol in solutions]}
                else:
                    steps.append("No solutions found.")
                    result["solution"] = "No solutions"
                
                result["steps"] = steps
                result["explanation"] = f"Solved the equation for {symbol_to_solve} by isolating the variable."
                
            else:
                # Not an equation, try to simplify
                result = self._solve_algebraic_expression(problem, expressions)
        
        except Exception as e:
            logger.error(f"Error solving algebraic equation: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_algebraic_expression(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve algebraic expressions (simplify, expand, factor).
        
        Args:
            problem: The algebraic expression problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract the expression
            expression = expressions[0]
            
            # Clean up the expression
            expression = expression.replace('^', '**')
            
            # Convert to SymPy expression
            expr = parse_expr(expression)
            
            # Determine operation based on problem text
            problem_lower = problem.lower()
            
            steps = []
            steps.append(f"Original expression: {expression}")
            
            if 'simplify' in problem_lower:
                result_expr = simplify(expr)
                steps.append(f"Simplify: {result_expr}")
                operation = "simplification"
            elif 'expand' in problem_lower:
                result_expr = expand(expr)
                steps.append(f"Expand: {result_expr}")
                operation = "expansion"
            elif 'factor' in problem_lower:
                result_expr = factor(expr)
                steps.append(f"Factor: {result_expr}")
                operation = "factorization"
            else:
                # Default to simplify
                result_expr = simplify(expr)
                steps.append(f"Simplify: {result_expr}")
                operation = "simplification"
            
            result["solution"] = str(result_expr)
            result["steps"] = steps
            result["explanation"] = f"Performed {operation} on the algebraic expression."
            
        except Exception as e:
            logger.error(f"Error solving algebraic expression: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_differentiation(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve differentiation problems.
        
        Args:
            problem: The differentiation problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract the expression
            expression = expressions[0]
            
            # Clean up the expression
            expression = expression.replace('^', '**')
            
            # Determine the variable to differentiate with respect to
            # Default to x if not specified
            var_match = re.search(r'd/d([a-zA-Z])', problem)
            if var_match:
                var_name = var_match.group(1)
            else:
                var_name = 'x'
            
            # Create the symbol
            var = symbols(var_name)
            
            # Convert to SymPy expression
            expr = parse_expr(expression, local_dict={var_name: var})
            
            # Add steps
            steps = []
            steps.append(f"Original expression: {expression}")
            steps.append(f"Differentiate with respect to {var_name}")
            
            # Differentiate
            derivative = diff(expr, var)
            
            # Add the result to steps
            steps.append(f"Apply differentiation rules")
            steps.append(f"Result: {derivative}")
            
            # Simplify if possible
            simplified = simplify(derivative)
            if simplified != derivative:
                steps.append(f"Simplify: {simplified}")
                derivative = simplified
            
            result["solution"] = str(derivative)
            result["steps"] = steps
            result["explanation"] = f"Differentiated the expression with respect to {var_name} using calculus rules."
            
        except Exception as e:
            logger.error(f"Error solving differentiation problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_integration(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve integration problems.
        
        Args:
            problem: The integration problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract the expression
            expression = expressions[0]
            
            # Clean up the expression
            expression = expression.replace('^', '**')
            
            # Determine the variable to integrate with respect to
            # Default to x if not specified
            var_match = re.search(r'\\int_\{[^}]*\}\^\{[^}]*\}.*?d([a-zA-Z])', problem) or \
                       re.search(r'\\int.*?d([a-zA-Z])', problem) or \
                       re.search(r'∫.*?d([a-zA-Z])', problem) or \
                       re.search(r'integrate.*with respect to\s+([a-zA-Z])', problem)
            
            if var_match:
                var_name = var_match.group(1)
            else:
                var_name = 'x'
            
            # Create the symbol
            var = symbols(var_name)
            
            # Convert to SymPy expression
            expr = parse_expr(expression, local_dict={var_name: var})
            
            # Check for definite integral
            limits_match = re.search(r'from\s+([\d\.-]+)\s+to\s+([\d\.-]+)', problem) or \
                          re.search(r'\\int_\{([\d\.-]+)\}\^\{([\d\.-]+)\}', problem) or \
                          re.search(r'∫_\{([\d\.-]+)\}\^\{([\d\.-]+)\}', problem)
            
            # Add steps
            steps = []
            steps.append(f"Original expression: {expression}")
            
            if limits_match:
                # Definite integral
                lower_limit = float(limits_match.group(1))
                upper_limit = float(limits_match.group(2))
                
                steps.append(f"Integrate with respect to {var_name} from {lower_limit} to {upper_limit}")
                
                # Integrate with limits
                integral = integrate(expr, (var, lower_limit, upper_limit))
                
                steps.append(f"Apply integration rules")
                steps.append(f"Evaluate from {lower_limit} to {upper_limit}")
                steps.append(f"Result: {integral}")
                
            else:
                # Indefinite integral
                steps.append(f"Integrate with respect to {var_name}")
                
                # Integrate
                integral = integrate(expr, var)
                
                steps.append(f"Apply integration rules")
                steps.append(f"Result: {integral} + C")
                
                # Add constant of integration
                integral_str = f"{integral} + C"
                integral = integral_str
            
            result["solution"] = str(integral)
            result["steps"] = steps
            result["explanation"] = f"Integrated the expression with respect to {var_name} using calculus rules."
            
        except Exception as e:
            logger.error(f"Error solving integration problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_linear_algebra(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve linear algebra problems.
        
        Args:
            problem: The linear algebra problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            problem_lower = problem.lower()
            
            # Determine the type of linear algebra problem
            if 'determinant' in problem_lower:
                result = self._solve_determinant(problem, expressions)
            elif 'eigenvalue' in problem_lower or 'eigenvector' in problem_lower:
                result = self._solve_eigenvalue(problem, expressions)
            elif 'inverse' in problem_lower:
                result = self._solve_matrix_inverse(problem, expressions)
            elif 'system of equation' in problem_lower or 'linear system' in problem_lower:
                result = self._solve_linear_system(problem, expressions)
            elif 'dot product' in problem_lower:
                result = self._solve_dot_product(problem, expressions)
            elif 'cross product' in problem_lower:
                result = self._solve_cross_product(problem, expressions)
            else:
                # Default to matrix operations
                result = self._solve_matrix_operations(problem, expressions)
            
        except Exception as e:
            logger.error(f"Error solving linear algebra problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_determinant(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve matrix determinant problems.
        
        Args:
            problem: The determinant problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract matrix from the problem
            # This is a simplified approach - in a real implementation,
            # you would need more sophisticated parsing
            matrix_str = expressions[0]
            
            # Parse the matrix string into a SymPy Matrix
            # Assuming matrix is in the format [[a,b],[c,d]]
            matrix_str = matrix_str.replace('[', '').replace(']', '')
            rows = matrix_str.split(';')
            matrix_data = []
            
            for row in rows:
                row_values = [float(val.strip()) for val in row.split(',')]
                matrix_data.append(row_values)
            
            matrix = Matrix(matrix_data)
            
            # Add steps
            steps = []
            steps.append(f"Matrix: {matrix}")
            
            # Calculate determinant
            det = matrix.det()
            
            steps.append(f"Calculate determinant")
            steps.append(f"Result: {det}")
            
            result["solution"] = float(det)
            result["steps"] = steps
            result["explanation"] = "Calculated the determinant of the matrix."
            
        except Exception as e:
            logger.error(f"Error solving determinant problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_eigenvalue(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve eigenvalue/eigenvector problems.
        
        Args:
            problem: The eigenvalue problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract matrix from the problem
            matrix_str = expressions[0]
            
            # Parse the matrix string into a SymPy Matrix
            matrix_str = matrix_str.replace('[', '').replace(']', '')
            rows = matrix_str.split(';')
            matrix_data = []
            
            for row in rows:
                row_values = [float(val.strip()) for val in row.split(',')]
                matrix_data.append(row_values)
            
            matrix = Matrix(matrix_data)
            
            # Add steps
            steps = []
            steps.append(f"Matrix: {matrix}")
            
            # Calculate eigenvalues and eigenvectors
            eigenvals = matrix.eigenvals()
            eigenvects = matrix.eigenvects()
            
            steps.append(f"Calculate eigenvalues and eigenvectors")
            steps.append(f"Eigenvalues: {eigenvals}")
            steps.append(f"Eigenvectors: {eigenvects}")
            
            # Format the solution
            eigenvalue_dict = {}
            for val, mult in eigenvals.items():
                eigenvalue_dict[str(val)] = {
                    "multiplicity": mult,
                    "eigenvectors": []
                }
            
            for val, mult, basis in eigenvects:
                for vec in basis:
                    eigenvalue_dict[str(val)]["eigenvectors"].append(str(vec))
            
            result["solution"] = eigenvalue_dict
            result["steps"] = steps
            result["explanation"] = "Calculated the eigenvalues and eigenvectors of the matrix."
            
        except Exception as e:
            logger.error(f"Error solving eigenvalue problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_matrix_inverse(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve matrix inverse problems.
        
        Args:
            problem: The matrix inverse problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract matrix from the problem
            matrix_str = expressions[0]
            
            # Parse the matrix string into a SymPy Matrix
            matrix_str = matrix_str.replace('[', '').replace(']', '')
            rows = matrix_str.split(';')
            matrix_data = []
            
            for row in rows:
                row_values = [float(val.strip()) for val in row.split(',')]
                matrix_data.append(row_values)
            
            matrix = Matrix(matrix_data)
            
            # Add steps
            steps = []
            steps.append(f"Matrix: {matrix}")
            
            # Check if matrix is invertible
            det = matrix.det()
            steps.append(f"Check if determinant is non-zero: det = {det}")
            
            if det == 0:
                steps.append("Matrix is not invertible (determinant is zero)")
                result["solution"] = "Not invertible"
            else:
                # Calculate inverse
                inverse = matrix.inv()
                
                steps.append(f"Calculate inverse")
                steps.append(f"Result: {inverse}")
                
                # Convert to list of lists for easier serialization
                inverse_list = []
                for i in range(inverse.rows):
                    row = []
                    for j in range(inverse.cols):
                        row.append(float(inverse[i, j]))
                    inverse_list.append(row)
                
                result["solution"] = inverse_list
            
            result["steps"] = steps
            result["explanation"] = "Calculated the inverse of the matrix using the adjugate method."
            
        except Exception as e:
            logger.error(f"Error solving matrix inverse problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_linear_system(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve systems of linear equations.
        
        Args:
            problem: The linear system problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # For a system of equations, we need to parse multiple equations
            # This is a simplified approach
            equations = []
            variables = set()
            
            for expr in expressions:
                if '=' in expr:
                    left, right = expr.split('=', 1)
                    eq = parse_expr(left.strip() + '-(' + right.strip() + ')')
                    equations.append(eq)
                    variables.update(eq.free_symbols)
            
            if not equations:
                result["error"] = "No equations found in the problem."
                return result
            
            # Convert variables to a sorted list for consistent ordering
            variables = sorted(list(variables), key=lambda x: str(x))
            
            # Add steps
            steps = []
            steps.append(f"System of equations:")
            for i, eq in enumerate(equations):
                steps.append(f"Equation {i+1}: {eq} = 0")
            
            steps.append(f"Variables: {', '.join(str(var) for var in variables)}")
            
            # Solve the system
            solution = solve(equations, variables, dict=True)
            
            if solution:
                steps.append(f"Solution:")
                for sol in solution:
                    for var, val in sol.items():
                        steps.append(f"{var} = {val}")
                
                # Format the solution
                formatted_solution = []
                for sol in solution:
                    sol_dict = {}
                    for var, val in sol.items():
                        sol_dict[str(var)] = str(val)
                    formatted_solution.append(sol_dict)
                
                result["solution"] = formatted_solution
            else:
                steps.append("No solution found.")
                result["solution"] = "No solution"
            
            result["steps"] = steps
            result["explanation"] = "Solved the system of linear equations using elimination and substitution methods."
            
        except Exception as e:
            logger.error(f"Error solving linear system: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_dot_product(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve dot product problems.
        
        Args:
            problem: The dot product problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract vectors from the problem
            # Assuming format like "[1,2,3] dot [4,5,6]"
            vectors_match = re.search(r'\[([-\d\s,\.]+)\]\s*(?:dot|\.)\s*\[([-\d\s,\.]+)\]', problem)
            
            if not vectors_match:
                result["error"] = "Could not parse vectors from the problem."
                return result
            
            # Parse vectors
            vector1_str = vectors_match.group(1)
            vector2_str = vectors_match.group(2)
            
            vector1 = [float(x.strip()) for x in vector1_str.split(',')]
            vector2 = [float(x.strip()) for x in vector2_str.split(',')]
            
            # Check if vectors have the same dimension
            if len(vector1) != len(vector2):
                result["error"] = "Vectors must have the same dimension for dot product."
                return result
            
            # Add steps
            steps = []
            steps.append(f"Vector 1: {vector1}")
            steps.append(f"Vector 2: {vector2}")
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            
            steps.append(f"Calculate dot product: {' + '.join(f'{a} * {b}' for a, b in zip(vector1, vector2))}")
            steps.append(f"Result: {dot_product}")
            
            result["solution"] = dot_product
            result["steps"] = steps
            result["explanation"] = "Calculated the dot product of the two vectors by multiplying corresponding components and summing the results."
            
        except Exception as e:
            logger.error(f"Error solving dot product problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_cross_product(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve cross product problems.
        
        Args:
            problem: The cross product problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # Extract vectors from the problem
            # Assuming format like "[1,2,3] cross [4,5,6]"
            vectors_match = re.search(r'\[([-\d\s,\.]+)\]\s*(?:cross|×)\s*\[([-\d\s,\.]+)\]', problem)
            
            if not vectors_match:
                result["error"] = "Could not parse vectors from the problem."
                return result
            
            # Parse vectors
            vector1_str = vectors_match.group(1)
            vector2_str = vectors_match.group(2)
            
            vector1 = [float(x.strip()) for x in vector1_str.split(',')]
            vector2 = [float(x.strip()) for x in vector2_str.split(',')]
            
            # Check if vectors are 3D
            if len(vector1) != 3 or len(vector2) != 3:
                result["error"] = "Cross product is defined only for 3D vectors."
                return result
            
            # Add steps
            steps = []
            steps.append(f"Vector 1: {vector1}")
            steps.append(f"Vector 2: {vector2}")
            
            # Calculate cross product
            cross_product = [
                vector1[1] * vector2[2] - vector1[2] * vector2[1],
                vector1[2] * vector2[0] - vector1[0] * vector2[2],
                vector1[0] * vector2[1] - vector1[1] * vector2[0]
            ]
            
            steps.append(f"Calculate cross product using the formula:")
            steps.append(f"i: {vector1[1]} * {vector2[2]} - {vector1[2]} * {vector2[1]} = {cross_product[0]}")
            steps.append(f"j: {vector1[2]} * {vector2[0]} - {vector1[0]} * {vector2[2]} = {cross_product[1]}")
            steps.append(f"k: {vector1[0]} * {vector2[1]} - {vector1[1]} * {vector2[0]} = {cross_product[2]}")
            steps.append(f"Result: [{cross_product[0]}, {cross_product[1]}, {cross_product[2]}]")
            
            result["solution"] = cross_product
            result["steps"] = steps
            result["explanation"] = "Calculated the cross product of the two 3D vectors using the determinant formula."
            
        except Exception as e:
            logger.error(f"Error solving cross product problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_matrix_operations(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve general matrix operations.
        
        Args:
            problem: The matrix operation problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            # This is a placeholder for general matrix operations
            # In a real implementation, you would need to parse the problem
            # to determine the specific operation and matrices involved
            
            result["error"] = "General matrix operations not implemented yet."
            
        except Exception as e:
            logger.error(f"Error solving matrix operation: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_statistics(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve statistics problems.
        
        Args:
            problem: The statistics problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            problem_lower = problem.lower()
            
            # Extract data from the problem
            # Look for numbers in the problem text
            data_match = re.findall(r'\d+\.?\d*', problem)
            data = [float(x) for x in data_match]
            
            if not data:
                result["error"] = "No numerical data found in the problem."
                return result
            
            # Add steps
            steps = []
            steps.append(f"Data: {data}")
            
            # Determine the type of statistics problem
            if 'mean' in problem_lower or 'average' in problem_lower:
                # Calculate mean
                mean = sum(data) / len(data)
                steps.append(f"Calculate mean: sum({data}) / {len(data)} = {mean}")
                result["solution"] = mean
                result["explanation"] = "Calculated the mean (average) of the data set."
                
            elif 'median' in problem_lower:
                # Calculate median
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                else:
                    median = sorted_data[n//2]
                
                steps.append(f"Sort data: {sorted_data}")
                steps.append(f"Calculate median: {median}")
                result["solution"] = median
                result["explanation"] = "Calculated the median of the data set."
                
            elif 'mode' in problem_lower:
                # Calculate mode
                from collections import Counter
                counter = Counter(data)
                mode_count = max(counter.values())
                modes = [k for k, v in counter.items() if v == mode_count]
                
                steps.append(f"Count occurrences: {dict(counter)}")
                steps.append(f"Find most frequent value(s): {modes}")
                result["solution"] = modes
                result["explanation"] = "Calculated the mode(s) of the data set."
                
            elif 'standard deviation' in problem_lower or 'std' in problem_lower:
                # Calculate standard deviation
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                std_dev = variance ** 0.5
                
                steps.append(f"Calculate mean: {mean}")
                steps.append(f"Calculate variance: {variance}")
                steps.append(f"Calculate standard deviation: sqrt({variance}) = {std_dev}")
                result["solution"] = std_dev
                result["explanation"] = "Calculated the standard deviation of the data set."
                
            elif 'variance' in problem_lower:
                # Calculate variance
                mean = sum(data) / len(data)
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                
                steps.append(f"Calculate mean: {mean}")
                steps.append(f"Calculate variance: {variance}")
                result["solution"] = variance
                result["explanation"] = "Calculated the variance of the data set."
                
            else:
                # Default to basic statistics
                mean = sum(data) / len(data)
                sorted_data = sorted(data)
                n = len(sorted_data)
                if n % 2 == 0:
                    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
                else:
                    median = sorted_data[n//2]
                
                variance = sum((x - mean) ** 2 for x in data) / len(data)
                std_dev = variance ** 0.5
                
                steps.append(f"Calculate mean: {mean}")
                steps.append(f"Calculate median: {median}")
                steps.append(f"Calculate variance: {variance}")
                steps.append(f"Calculate standard deviation: {std_dev}")
                
                result["solution"] = {
                    "mean": mean,
                    "median": median,
                    "variance": variance,
                    "standard_deviation": std_dev
                }
                result["explanation"] = "Calculated basic statistical measures for the data set."
            
            result["steps"] = steps
            
        except Exception as e:
            logger.error(f"Error solving statistics problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_geometry(self, problem: str, expressions: List[str]) -> Dict[str, Any]:
        """
        Solve geometry problems.
        
        Args:
            problem: The geometry problem
            expressions: Extracted expressions
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        try:
            problem_lower = problem.lower()
            
            # Extract numbers from the problem
            numbers = re.findall(r'\d+\.?\d*', problem)
            values = [float(x) for x in numbers]
            
            # Add steps
            steps = []
            
            # Determine the type of geometry problem
            if 'area' in problem_lower and 'triangle' in problem_lower:
                # Area of a triangle
                if len(values) >= 2:
                    base = values[0]
                    height = values[1]
                    area = 0.5 * base * height
                    
                    steps.append(f"Triangle with base = {base} and height = {height}")
                    steps.append(f"Calculate area: 0.5 * {base} * {height} = {area}")
                    result["solution"] = area
                    result["explanation"] = "Calculated the area of a triangle using the formula: Area = 0.5 * base * height."
                else:
                    result["error"] = "Insufficient values for triangle area calculation."
                    
            elif 'area' in problem_lower and 'circle' in problem_lower:
                # Area of a circle
                if len(values) >= 1:
                    radius = values[0]
                    area = np.pi * radius ** 2
                    
                    steps.append(f"Circle with radius = {radius}")
                    steps.append(f"Calculate area: π * {radius}² = {area}")
                    result["solution"] = area
                    result["explanation"] = "Calculated the area of a circle using the formula: Area = π * radius²."
                else:
                    result["error"] = "Insufficient values for circle area calculation."
                    
            elif 'area' in problem_lower and ('rectangle' in problem_lower or 'square' in problem_lower):
                # Area of a rectangle or square
                if 'square' in problem_lower and len(values) >= 1:
                    side = values[0]
                    area = side ** 2
                    
                    steps.append(f"Square with side = {side}")
                    steps.append(f"Calculate area: {side}² = {area}")
                    result["solution"] = area
                    result["explanation"] = "Calculated the area of a square using the formula: Area = side²."
                elif len(values) >= 2:
                    length = values[0]
                    width = values[1]
                    area = length * width
                    
                    steps.append(f"Rectangle with length = {length} and width = {width}")
                    steps.append(f"Calculate area: {length} * {width} = {area}")
                    result["solution"] = area
                    result["explanation"] = "Calculated the area of a rectangle using the formula: Area = length * width."
                else:
                    result["error"] = "Insufficient values for rectangle/square area calculation."
                    
            elif 'perimeter' in problem_lower and 'triangle' in problem_lower:
                # Perimeter of a triangle
                if len(values) >= 3:
                    a, b, c = values[:3]
                    perimeter = a + b + c
                    
                    steps.append(f"Triangle with sides = {a}, {b}, {c}")
                    steps.append(f"Calculate perimeter: {a} + {b} + {c} = {perimeter}")
                    result["solution"] = perimeter
                    result["explanation"] = "Calculated the perimeter of a triangle by summing all sides."
                else:
                    result["error"] = "Insufficient values for triangle perimeter calculation."
                    
            elif 'perimeter' in problem_lower and 'circle' in problem_lower:
                # Circumference of a circle
                if len(values) >= 1:
                    radius = values[0]
                    circumference = 2 * np.pi * radius
                    
                    steps.append(f"Circle with radius = {radius}")
                    steps.append(f"Calculate circumference: 2 * π * {radius} = {circumference}")
                    result["solution"] = circumference
                    result["explanation"] = "Calculated the circumference of a circle using the formula: Circumference = 2 * π * radius."
                else:
                    result["error"] = "Insufficient values for circle circumference calculation."
                    
            elif 'perimeter' in problem_lower and ('rectangle' in problem_lower or 'square' in problem_lower):
                # Perimeter of a rectangle or square
                if 'square' in problem_lower and len(values) >= 1:
                    side = values[0]
                    perimeter = 4 * side
                    
                    steps.append(f"Square with side = {side}")
                    steps.append(f"Calculate perimeter: 4 * {side} = {perimeter}")
                    result["solution"] = perimeter
                    result["explanation"] = "Calculated the perimeter of a square using the formula: Perimeter = 4 * side."
                elif len(values) >= 2:
                    length = values[0]
                    width = values[1]
                    perimeter = 2 * (length + width)
                    
                    steps.append(f"Rectangle with length = {length} and width = {width}")
                    steps.append(f"Calculate perimeter: 2 * ({length} + {width}) = {perimeter}")
                    result["solution"] = perimeter
                    result["explanation"] = "Calculated the perimeter of a rectangle using the formula: Perimeter = 2 * (length + width)."
                else:
                    result["error"] = "Insufficient values for rectangle/square perimeter calculation."
                    
            elif 'volume' in problem_lower and 'sphere' in problem_lower:
                # Volume of a sphere
                if len(values) >= 1:
                    radius = values[0]
                    volume = (4/3) * np.pi * radius ** 3
                    
                    steps.append(f"Sphere with radius = {radius}")
                    steps.append(f"Calculate volume: (4/3) * π * {radius}³ = {volume}")
                    result["solution"] = volume
                    result["explanation"] = "Calculated the volume of a sphere using the formula: Volume = (4/3) * π * radius³."
                else:
                    result["error"] = "Insufficient values for sphere volume calculation."
                    
            elif 'volume' in problem_lower and 'cube' in problem_lower:
                # Volume of a cube
                if len(values) >= 1:
                    side = values[0]
                    volume = side ** 3
                    
                    steps.append(f"Cube with side = {side}")
                    steps.append(f"Calculate volume: {side}³ = {volume}")
                    result["solution"] = volume
                    result["explanation"] = "Calculated the volume of a cube using the formula: Volume = side³."
                else:
                    result["error"] = "Insufficient values for cube volume calculation."
                    
            elif 'volume' in problem_lower and 'cylinder' in problem_lower:
                # Volume of a cylinder
                if len(values) >= 2:
                    radius = values[0]
                    height = values[1]
                    volume = np.pi * radius ** 2 * height
                    
                    steps.append(f"Cylinder with radius = {radius} and height = {height}")
                    steps.append(f"Calculate volume: π * {radius}² * {height} = {volume}")
                    result["solution"] = volume
                    result["explanation"] = "Calculated the volume of a cylinder using the formula: Volume = π * radius² * height."
                else:
                    result["error"] = "Insufficient values for cylinder volume calculation."
                    
            elif 'pythagorean' in problem_lower or ('triangle' in problem_lower and 'right' in problem_lower):
                # Pythagorean theorem
                if len(values) >= 2:
                    a = values[0]
                    b = values[1]
                    c = (a ** 2 + b ** 2) ** 0.5
                    
                    steps.append(f"Right triangle with sides a = {a} and b = {b}")
                    steps.append(f"Calculate hypotenuse using Pythagorean theorem: c = √(a² + b²) = √({a}² + {b}²) = {c}")
                    result["solution"] = c
                    result["explanation"] = "Calculated the hypotenuse of a right triangle using the Pythagorean theorem: c = √(a² + b²)."
                else:
                    result["error"] = "Insufficient values for Pythagorean theorem calculation."
                    
            else:
                # Use LLM for other geometry problems
                if self.llm:
                    result = self._solve_with_llm(problem, None)
                else:
                    result["error"] = "Specific geometry problem type not recognized."
            
            result["steps"] = steps
            
        except Exception as e:
            logger.error(f"Error solving geometry problem: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _solve_with_llm(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve a problem using the language model.
        
        Args:
            problem: The problem to solve
            context: Additional context for the problem
            
        Returns:
            Dict[str, Any]: Solution data
        """
        result = {
            "solution": None,
            "steps": [],
            "explanation": ""
        }
        
        if not self.llm:
            result["error"] = "Language model not available for solving this problem type."
            return result
        
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for problem-solving
            messages = [
                SystemMessage(content="You are an expert mathematical problem solver. "
                                    "Solve the following problem step by step, showing all your work. "
                                    "Include clear explanations for each step."),
                HumanMessage(content=f"Problem: {problem}")
            ]
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Parse the solution
            # Extract steps and final answer
            steps = []
            explanation = ""
            solution = None
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            lines = solution_text.split('\n')
            for line in lines:
                if line.strip():
                    steps.append(line.strip())
            
            # Try to extract the final answer
            answer_patterns = [
                r'final answer:?\s*(.+)',
                r'answer:?\s*(.+)',
                r'solution:?\s*(.+)',
                r'result:?\s*(.+)',
                r'therefore,?\s*(.+)'
            ]
            
            for pattern in answer_patterns:
                for line in reversed(lines):  # Start from the end
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        solution = match.group(1).strip()
                        break
                if solution:
                    break
            
            # If no clear answer found, use the last line
            if not solution and steps:
                solution = steps[-1]
            
            # Generate explanation
            explanation = "Solution generated using advanced mathematical reasoning and problem-solving techniques."
            
            result["solution"] = solution
            result["steps"] = steps
            result["explanation"] = explanation
            
        except Exception as e:
            logger.error(f"Error solving with LLM: {str(e)}")
            result["error"] = str(e)
        
        return result
    
    def _verify_equation_solution(self, expressions: List[str], solution: Any) -> bool:
        """
        Verify a solution to an equation by substituting it back.
        
        Args:
            expressions: The equation expressions
            solution: The solution to verify
            
        Returns:
            bool: Whether the solution is correct
        """
        try:
            # Extract the equation
            equation = expressions[0]
            
            # Check if it's an equation (contains =)
            if '=' in equation:
                # Split into left and right sides
                left_side, right_side = equation.split('=', 1)
                
                # Convert to SymPy expressions
                left_expr = parse_expr(left_side.strip())
                right_expr = parse_expr(right_side.strip())
                
                # Find all symbols (variables)
                symbols_in_expr = list(left_expr.free_symbols.union(right_expr.free_symbols))
                
                if not symbols_in_expr:
                    return False
                
                # Get the symbol to verify
                symbol_to_verify = symbols_in_expr[0]
                
                # Convert solution to appropriate type
                if isinstance(solution, dict):
                    # Solution is in the format {"x": ["1", "2"]}
                    symbol_name = str(symbol_to_verify)
                    if symbol_name in solution:
                        solutions_to_check = solution[symbol_name]
                        if not isinstance(solutions_to_check, list):
                            solutions_to_check = [solutions_to_check]
                    else:
                        return False
                elif isinstance(solution, list):
                    # Solution is a list of values
                    solutions_to_check = solution
                else:
                    # Solution is a single value
                    solutions_to_check = [solution]
                
                # Verify each solution
                for sol in solutions_to_check:
                    # Convert to float if possible
                    try:
                        sol_value = float(sol)
                    except (ValueError, TypeError):
                        sol_value = sol
                    
                    # Substitute the solution
                    left_result = left_expr.subs(symbol_to_verify, sol_value)
                    right_result = right_expr.subs(symbol_to_verify, sol_value)
                    
                    # Check if the equation is satisfied (with some tolerance for floating-point errors)
                    if not abs(float(left_result) - float(right_result)) < 1e-10:
                        return False
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error verifying equation solution: {str(e)}")
            return False
    
    def _compare_solutions(self, solution1: Any, solution2: Any) -> bool:
        """
        Compare two solutions for equality.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            bool: Whether the solutions are equal
        """
        try:
            # Convert to float if possible
            try:
                sol1 = float(solution1)
                sol2 = float(solution2)
                return abs(sol1 - sol2) < 1e-10
            except (ValueError, TypeError):
                pass
            
            # Compare strings
            if isinstance(solution1, str) and isinstance(solution2, str):
                return solution1.strip() == solution2.strip()
            
            # Compare lists
            if isinstance(solution1, list) and isinstance(solution2, list):
                if len(solution1) != len(solution2):
                    return False
                
                for s1, s2 in zip(solution1, solution2):
                    if not self._compare_solutions(s1, s2):
                        return False
                
                return True
            
            # Compare dictionaries
            if isinstance(solution1, dict) and isinstance(solution2, dict):
                if set(solution1.keys()) != set(solution2.keys()):
                    return False
                
                for key in solution1:
                    if not self._compare_solutions(solution1[key], solution2[key]):
                        return False
                
                return True
            
            # Default comparison
            return solution1 == solution2
            
        except Exception as e:
            logger.error(f"Error comparing solutions: {str(e)}")
            return False
    
    def _compare_expressions(self, expr1: str, expr2: str) -> bool:
        """
        Compare two mathematical expressions for equality.
        
        Args:
            expr1: First expression
            expr2: Second expression
            
        Returns:
            bool: Whether the expressions are equal
        """
        try:
            # Convert to SymPy expressions
            sympy_expr1 = parse_expr(expr1)
            sympy_expr2 = parse_expr(expr2)
            
            # Check if the difference simplifies to zero
            diff = simplify(sympy_expr1 - sympy_expr2)
            return diff == 0
            
        except Exception as e:
            logger.error(f"Error comparing expressions: {str(e)}")
            return False