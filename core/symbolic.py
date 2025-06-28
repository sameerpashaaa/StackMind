import logging
import os
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from IPython.display import display, Math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SymbolicComputation:
    """Symbolic computation module for the AI Problem Solver.
    
    This module provides capabilities for symbolic mathematics and equation solving:
    - Symbolic algebra (simplification, expansion, factorization)
    - Equation solving (linear, quadratic, systems of equations)
    - Calculus (differentiation, integration, limits)
    - Linear algebra (matrices, determinants, eigenvalues)
    - Plotting mathematical functions
    - LaTeX rendering of mathematical expressions
    
    It uses SymPy for symbolic mathematics and Matplotlib for plotting.
    """
    
    def __init__(self):
        """Initialize the symbolic computation module."""
        # Define common symbols for convenience
        self.x, self.y, self.z = sp.symbols('x y z')
        self.t = sp.Symbol('t')
        self.n = sp.Symbol('n', integer=True)
        self.a, self.b, self.c = sp.symbols('a b c')
        
        # Define common functions
        self.f = sp.Function('f')
        self.g = sp.Function('g')
        
        # Store the last result for convenience
        self.last_result = None
    
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """Parse a string into a SymPy expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            
        Returns:
            SymPy expression.
            
        Raises:
            ValueError: If the expression cannot be parsed.
        """
        try:
            # Replace common mathematical notations with SymPy-compatible ones
            expr_str = expr_str.replace('^', '**')  # Replace caret with power operator
            
            # Parse the expression
            expr = sp.sympify(expr_str)
            return expr
        except Exception as e:
            logger.error(f"Error parsing expression '{expr_str}': {str(e)}")
            raise ValueError(f"Could not parse expression: {str(e)}")
    
    def parse_equation(self, eq_str: str) -> sp.Eq:
        """Parse a string into a SymPy equation.
        
        Args:
            eq_str: String representation of a mathematical equation.
            
        Returns:
            SymPy equation.
            
        Raises:
            ValueError: If the equation cannot be parsed.
        """
        try:
            # Check if the equation contains an equals sign
            if '=' not in eq_str:
                raise ValueError("Equation must contain an equals sign (=).")
            
            # Split the equation into left and right sides
            left_str, right_str = eq_str.split('=', 1)
            
            # Parse both sides
            left_expr = self.parse_expression(left_str)
            right_expr = self.parse_expression(right_str)
            
            # Create the equation
            eq = sp.Eq(left_expr, right_expr)
            return eq
        except Exception as e:
            logger.error(f"Error parsing equation '{eq_str}': {str(e)}")
            raise ValueError(f"Could not parse equation: {str(e)}")
    
    def simplify(self, expr_str: str) -> Dict[str, Any]:
        """Simplify a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            
        Returns:
            Dictionary containing the original expression, simplified expression,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Simplify the expression
            simplified = sp.simplify(expr)
            self.last_result = simplified
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "simplified": {
                    "string": str(simplified),
                    "latex": sp.latex(simplified)
                },
                "type": "simplification"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error simplifying expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def expand(self, expr_str: str) -> Dict[str, Any]:
        """Expand a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            
        Returns:
            Dictionary containing the original expression, expanded expression,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Expand the expression
            expanded = sp.expand(expr)
            self.last_result = expanded
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "expanded": {
                    "string": str(expanded),
                    "latex": sp.latex(expanded)
                },
                "type": "expansion"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error expanding expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def factor(self, expr_str: str) -> Dict[str, Any]:
        """Factor a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            
        Returns:
            Dictionary containing the original expression, factored expression,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Factor the expression
            factored = sp.factor(expr)
            self.last_result = factored
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "factored": {
                    "string": str(factored),
                    "latex": sp.latex(factored)
                },
                "type": "factorization"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error factoring expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def solve_equation(self, eq_str: str, var_str: Optional[str] = None) -> Dict[str, Any]:
        """Solve a mathematical equation.
        
        Args:
            eq_str: String representation of a mathematical equation.
            var_str: Variable to solve for. If None, the first symbol in the equation will be used.
            
        Returns:
            Dictionary containing the original equation, solutions,
            and LaTeX representations.
        """
        try:
            # Parse the equation
            eq = self.parse_equation(eq_str)
            
            # Determine the variable to solve for
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use the first symbol in the equation
                symbols = list(eq.free_symbols)
                if not symbols:
                    raise ValueError("No variables found in the equation.")
                var = symbols[0]
            
            # Solve the equation
            solutions = sp.solve(eq, var)
            self.last_result = solutions
            
            # Create the result
            result = {
                "original": {
                    "string": eq_str,
                    "latex": sp.latex(eq)
                },
                "variable": str(var),
                "solutions": [
                    {"string": str(sol), "latex": sp.latex(sol)}
                    for sol in solutions
                ],
                "type": "equation_solving"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error solving equation: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def solve_system(self, equations: List[str], variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """Solve a system of equations.
        
        Args:
            equations: List of string representations of mathematical equations.
            variables: List of variables to solve for. If None, all symbols in the equations will be used.
            
        Returns:
            Dictionary containing the original equations, solutions,
            and LaTeX representations.
        """
        try:
            # Parse the equations
            eqs = [self.parse_equation(eq_str) for eq_str in equations]
            
            # Determine the variables to solve for
            if variables:
                vars = [sp.Symbol(var) for var in variables]
            else:
                # Use all symbols in the equations
                vars = list(set().union(*[eq.free_symbols for eq in eqs]))
                if not vars:
                    raise ValueError("No variables found in the equations.")
            
            # Solve the system of equations
            solutions = sp.solve(eqs, vars)
            self.last_result = solutions
            
            # Create the result
            result = {
                "original": [
                    {"string": eq_str, "latex": sp.latex(eq)}
                    for eq_str, eq in zip(equations, eqs)
                ],
                "variables": [str(var) for var in vars],
                "type": "system_solving"
            }
            
            # Handle different solution formats
            if isinstance(solutions, list):
                # Multiple solution sets
                result["solutions"] = [
                    {
                        "values": [
                            {"variable": str(var), "value": str(sol[i]), "latex": sp.latex(sol[i])}
                            for i, var in enumerate(vars)
                        ]
                    }
                    for sol in solutions
                ]
            elif isinstance(solutions, dict):
                # Single solution set
                result["solutions"] = [
                    {
                        "values": [
                            {"variable": str(var), "value": str(solutions[var]), "latex": sp.latex(solutions[var])}
                            for var in vars if var in solutions
                        ]
                    }
                ]
            else:
                # Empty solution or other format
                result["solutions"] = [{"string": str(solutions), "latex": sp.latex(solutions)}]
            
            return result
        except Exception as e:
            logger.error(f"Error solving system of equations: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def differentiate(self, expr_str: str, var_str: Optional[str] = None, order: int = 1) -> Dict[str, Any]:
        """Differentiate a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_str: Variable to differentiate with respect to. If None, 'x' will be used.
            order: Order of differentiation.
            
        Returns:
            Dictionary containing the original expression, derivative,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variable to differentiate with respect to
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use 'x' as the default variable
                var = sp.Symbol('x')
                # Check if 'x' is in the expression
                if var not in expr.free_symbols and expr.free_symbols:
                    # Use the first symbol in the expression
                    var = list(expr.free_symbols)[0]
            
            # Differentiate the expression
            derivative = sp.diff(expr, var, order)
            self.last_result = derivative
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "variable": str(var),
                "order": order,
                "derivative": {
                    "string": str(derivative),
                    "latex": sp.latex(derivative)
                },
                "type": "differentiation"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error differentiating expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def integrate(self, expr_str: str, var_str: Optional[str] = None, 
                 limits: Optional[Tuple[Union[str, float], Union[str, float]]] = None) -> Dict[str, Any]:
        """Integrate a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_str: Variable to integrate with respect to. If None, 'x' will be used.
            limits: Tuple of (lower, upper) limits for definite integration.
                   If None, indefinite integration will be performed.
            
        Returns:
            Dictionary containing the original expression, integral,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variable to integrate with respect to
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use 'x' as the default variable
                var = sp.Symbol('x')
                # Check if 'x' is in the expression
                if var not in expr.free_symbols and expr.free_symbols:
                    # Use the first symbol in the expression
                    var = list(expr.free_symbols)[0]
            
            # Integrate the expression
            if limits:
                # Parse the limits
                lower = self.parse_expression(str(limits[0])) if isinstance(limits[0], str) else limits[0]
                upper = self.parse_expression(str(limits[1])) if isinstance(limits[1], str) else limits[1]
                
                # Perform definite integration
                integral = sp.integrate(expr, (var, lower, upper))
                integration_type = "definite_integration"
            else:
                # Perform indefinite integration
                integral = sp.integrate(expr, var)
                integration_type = "indefinite_integration"
            
            self.last_result = integral
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "variable": str(var),
                "integral": {
                    "string": str(integral),
                    "latex": sp.latex(integral)
                },
                "type": integration_type
            }
            
            # Add limits for definite integration
            if limits:
                result["limits"] = {
                    "lower": {"string": str(lower), "latex": sp.latex(lower)},
                    "upper": {"string": str(upper), "latex": sp.latex(upper)}
                }
            
            return result
        except Exception as e:
            logger.error(f"Error integrating expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_limit(self, expr_str: str, var_str: Optional[str] = None, 
                       limit_point: Union[str, float] = 0, direction: str = "") -> Dict[str, Any]:
        """Calculate the limit of a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_str: Variable to take the limit with respect to. If None, 'x' will be used.
            limit_point: Point at which to evaluate the limit.
            direction: Direction from which to approach the limit point ('+', '-', or '').
            
        Returns:
            Dictionary containing the original expression, limit,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variable to take the limit with respect to
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use 'x' as the default variable
                var = sp.Symbol('x')
                # Check if 'x' is in the expression
                if var not in expr.free_symbols and expr.free_symbols:
                    # Use the first symbol in the expression
                    var = list(expr.free_symbols)[0]
            
            # Parse the limit point
            if isinstance(limit_point, str):
                limit_point = self.parse_expression(limit_point)
            
            # Calculate the limit
            if direction == "+":
                limit = sp.limit(expr, var, limit_point, "+")
            elif direction == "-":
                limit = sp.limit(expr, var, limit_point, "-")
            else:
                limit = sp.limit(expr, var, limit_point)
            
            self.last_result = limit
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "variable": str(var),
                "limit_point": {"string": str(limit_point), "latex": sp.latex(limit_point)},
                "direction": direction,
                "limit": {
                    "string": str(limit),
                    "latex": sp.latex(limit)
                },
                "type": "limit"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating limit: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_series(self, expr_str: str, var_str: Optional[str] = None, 
                        about_point: Union[str, float] = 0, n_terms: int = 5) -> Dict[str, Any]:
        """Calculate the Taylor/Maclaurin series of a mathematical expression.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_str: Variable for the series expansion. If None, 'x' will be used.
            about_point: Point around which to expand the series.
            n_terms: Number of terms in the series expansion.
            
        Returns:
            Dictionary containing the original expression, series expansion,
            and LaTeX representations.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variable for the series expansion
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use 'x' as the default variable
                var = sp.Symbol('x')
                # Check if 'x' is in the expression
                if var not in expr.free_symbols and expr.free_symbols:
                    # Use the first symbol in the expression
                    var = list(expr.free_symbols)[0]
            
            # Parse the expansion point
            if isinstance(about_point, str):
                about_point = self.parse_expression(about_point)
            
            # Calculate the series expansion
            series = expr.series(var, about_point, n_terms).removeO()
            self.last_result = series
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "variable": str(var),
                "about_point": {"string": str(about_point), "latex": sp.latex(about_point)},
                "n_terms": n_terms,
                "series": {
                    "string": str(series),
                    "latex": sp.latex(series)
                },
                "type": "series_expansion"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating series expansion: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def solve_ode(self, eq_str: str, func_str: Optional[str] = None, 
                 var_str: Optional[str] = None, ics: Optional[Dict[str, Union[str, float]]] = None) -> Dict[str, Any]:
        """Solve an ordinary differential equation.
        
        Args:
            eq_str: String representation of a differential equation.
            func_str: Function to solve for. If None, 'y(x)' will be used.
            var_str: Independent variable. If None, 'x' will be used.
            ics: Initial conditions as a dictionary {"x0": value, "y0": value, "y'0": value, ...}.
            
        Returns:
            Dictionary containing the original equation, solution,
            and LaTeX representations.
        """
        try:
            # Determine the independent variable
            if var_str:
                var = sp.Symbol(var_str)
            else:
                var = sp.Symbol('x')
            
            # Determine the function to solve for
            if func_str:
                func = sp.Function(func_str)(var)
            else:
                func = sp.Function('y')(var)
            
            # Parse the equation
            # Replace derivatives with SymPy notation
            eq_str = eq_str.replace("y'", "Derivative(y(x), x)")
            eq_str = eq_str.replace("y''", "Derivative(y(x), x, 2)")
            eq_str = eq_str.replace("y'''", "Derivative(y(x), x, 3)")
            
            # Parse the equation
            eq = self.parse_equation(eq_str)
            
            # Solve the ODE
            if ics:
                # Parse initial conditions
                ics_dict = {}
                for key, value in ics.items():
                    if key.startswith("y") and "'" in key:
                        # Derivative initial condition
                        order = key.count("'")
                        ics_dict[sp.Derivative(func, var, order).subs(var, ics["x0"])] = value
                    elif key == "y0":
                        # Function value initial condition
                        ics_dict[func.subs(var, ics["x0"])] = value
                
                # Solve with initial conditions
                solution = sp.dsolve(eq, func, ics=ics_dict)
            else:
                # Solve without initial conditions
                solution = sp.dsolve(eq, func)
            
            self.last_result = solution
            
            # Create the result
            result = {
                "original": {
                    "string": eq_str,
                    "latex": sp.latex(eq)
                },
                "variable": str(var),
                "function": str(func),
                "solution": {
                    "string": str(solution),
                    "latex": sp.latex(solution)
                },
                "type": "ode_solving"
            }
            
            # Add initial conditions if provided
            if ics:
                result["initial_conditions"] = ics
            
            return result
        except Exception as e:
            logger.error(f"Error solving ODE: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_eigenvalues(self, matrix_str: str) -> Dict[str, Any]:
        """Calculate the eigenvalues and eigenvectors of a matrix.
        
        Args:
            matrix_str: String representation of a matrix.
            
        Returns:
            Dictionary containing the original matrix, eigenvalues, eigenvectors,
            and LaTeX representations.
        """
        try:
            # Parse the matrix
            matrix = self.parse_expression(matrix_str)
            
            # Check if the input is a matrix
            if not isinstance(matrix, sp.Matrix):
                raise ValueError("Input must be a matrix.")
            
            # Calculate eigenvalues and eigenvectors
            eigensystem = matrix.eigenvects()
            self.last_result = eigensystem
            
            # Create the result
            result = {
                "original": {
                    "string": matrix_str,
                    "latex": sp.latex(matrix)
                },
                "eigenvalues": [],
                "type": "eigenvalue_calculation"
            }
            
            # Format the eigenvalues and eigenvectors
            for eigenvalue, multiplicity, eigenvectors in eigensystem:
                ev_entry = {
                    "value": {"string": str(eigenvalue), "latex": sp.latex(eigenvalue)},
                    "multiplicity": multiplicity,
                    "eigenvectors": [
                        {"string": str(vec), "latex": sp.latex(vec)}
                        for vec in eigenvectors
                    ]
                }
                result["eigenvalues"].append(ev_entry)
            
            return result
        except Exception as e:
            logger.error(f"Error calculating eigenvalues: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_determinant(self, matrix_str: str) -> Dict[str, Any]:
        """Calculate the determinant of a matrix.
        
        Args:
            matrix_str: String representation of a matrix.
            
        Returns:
            Dictionary containing the original matrix, determinant,
            and LaTeX representations.
        """
        try:
            # Parse the matrix
            matrix = self.parse_expression(matrix_str)
            
            # Check if the input is a matrix
            if not isinstance(matrix, sp.Matrix):
                raise ValueError("Input must be a matrix.")
            
            # Calculate the determinant
            determinant = matrix.det()
            self.last_result = determinant
            
            # Create the result
            result = {
                "original": {
                    "string": matrix_str,
                    "latex": sp.latex(matrix)
                },
                "determinant": {
                    "string": str(determinant),
                    "latex": sp.latex(determinant)
                },
                "type": "determinant_calculation"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating determinant: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_inverse(self, matrix_str: str) -> Dict[str, Any]:
        """Calculate the inverse of a matrix.
        
        Args:
            matrix_str: String representation of a matrix.
            
        Returns:
            Dictionary containing the original matrix, inverse,
            and LaTeX representations.
        """
        try:
            # Parse the matrix
            matrix = self.parse_expression(matrix_str)
            
            # Check if the input is a matrix
            if not isinstance(matrix, sp.Matrix):
                raise ValueError("Input must be a matrix.")
            
            # Calculate the inverse
            inverse = matrix.inv()
            self.last_result = inverse
            
            # Create the result
            result = {
                "original": {
                    "string": matrix_str,
                    "latex": sp.latex(matrix)
                },
                "inverse": {
                    "string": str(inverse),
                    "latex": sp.latex(inverse)
                },
                "type": "inverse_calculation"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating inverse: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def plot_function(self, expr_str: str, var_str: Optional[str] = None, 
                     x_range: Tuple[float, float] = (-10, 10), 
                     y_range: Optional[Tuple[float, float]] = None, 
                     points: int = 1000, title: Optional[str] = None) -> Dict[str, Any]:
        """Plot a mathematical function.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_str: Variable for the function. If None, 'x' will be used.
            x_range: Range of x values to plot.
            y_range: Range of y values to plot. If None, it will be determined automatically.
            points: Number of points to plot.
            title: Title of the plot.
            
        Returns:
            Dictionary containing the original expression, plot data,
            and a base64-encoded image of the plot.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variable for the function
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use 'x' as the default variable
                var = sp.Symbol('x')
                # Check if 'x' is in the expression
                if var not in expr.free_symbols and expr.free_symbols:
                    # Use the first symbol in the expression
                    var = list(expr.free_symbols)[0]
            
            # Convert the expression to a numpy function
            f = sp.lambdify(var, expr, "numpy")
            
            # Generate x values
            x = np.linspace(x_range[0], x_range[1], points)
            
            # Calculate y values
            try:
                y = f(x)
                
                # Handle complex results
                if np.iscomplexobj(y):
                    y_real = np.real(y)
                    y_imag = np.imag(y)
                    has_complex = True
                else:
                    y_real = y
                    y_imag = None
                    has_complex = False
                
                # Create the plot
                plt.figure(figsize=(10, 6))
                
                if has_complex:
                    plt.subplot(2, 1, 1)
                    plt.plot(x, y_real, 'b-', label='Real Part')
                    if y_range:
                        plt.ylim(y_range)
                    plt.grid(True)
                    plt.legend()
                    plt.title(f"Real Part of {expr_str}" if not title else title)
                    
                    plt.subplot(2, 1, 2)
                    plt.plot(x, y_imag, 'r-', label='Imaginary Part')
                    if y_range:
                        plt.ylim(y_range)
                    plt.grid(True)
                    plt.legend()
                    plt.xlabel(str(var))
                    plt.title(f"Imaginary Part of {expr_str}")
                else:
                    plt.plot(x, y_real, 'b-')
                    if y_range:
                        plt.ylim(y_range)
                    plt.grid(True)
                    plt.xlabel(str(var))
                    plt.ylabel(f"{expr_str}")
                    plt.title(title if title else f"Plot of {expr_str}")
                
                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Encode the image as base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Close the plot to free memory
                plt.close()
                
                # Create the result
                result = {
                    "original": {
                        "string": expr_str,
                        "latex": sp.latex(expr)
                    },
                    "variable": str(var),
                    "x_range": list(x_range),
                    "has_complex": has_complex,
                    "plot_image": img_base64,
                    "type": "function_plot"
                }
                
                # Add y range if provided
                if y_range:
                    result["y_range"] = list(y_range)
                
                return result
            
            except Exception as e:
                logger.error(f"Error evaluating function: {str(e)}")
                return {"error": f"Error evaluating function: {str(e)}", "type": "error"}
        
        except Exception as e:
            logger.error(f"Error plotting function: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def plot_parametric(self, x_expr_str: str, y_expr_str: str, param_str: Optional[str] = None, 
                       t_range: Tuple[float, float] = (0, 2*np.pi), 
                       points: int = 1000, title: Optional[str] = None) -> Dict[str, Any]:
        """Plot a parametric curve.
        
        Args:
            x_expr_str: String representation of the x component.
            y_expr_str: String representation of the y component.
            param_str: Parameter variable. If None, 't' will be used.
            t_range: Range of parameter values to plot.
            points: Number of points to plot.
            title: Title of the plot.
            
        Returns:
            Dictionary containing the original expressions, plot data,
            and a base64-encoded image of the plot.
        """
        try:
            # Parse the expressions
            x_expr = self.parse_expression(x_expr_str)
            y_expr = self.parse_expression(y_expr_str)
            
            # Determine the parameter variable
            if param_str:
                param = sp.Symbol(param_str)
            else:
                param = sp.Symbol('t')
            
            # Convert the expressions to numpy functions
            x_func = sp.lambdify(param, x_expr, "numpy")
            y_func = sp.lambdify(param, y_expr, "numpy")
            
            # Generate parameter values
            t = np.linspace(t_range[0], t_range[1], points)
            
            # Calculate x and y values
            try:
                x = x_func(t)
                y = y_func(t)
                
                # Create the plot
                plt.figure(figsize=(8, 8))
                plt.plot(x, y, 'b-')
                plt.grid(True)
                plt.axis('equal')  # Equal aspect ratio
                plt.xlabel(f"{x_expr_str}")
                plt.ylabel(f"{y_expr_str}")
                plt.title(title if title else f"Parametric Plot: ({x_expr_str}, {y_expr_str})")
                
                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Encode the image as base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Close the plot to free memory
                plt.close()
                
                # Create the result
                result = {
                    "x_component": {
                        "string": x_expr_str,
                        "latex": sp.latex(x_expr)
                    },
                    "y_component": {
                        "string": y_expr_str,
                        "latex": sp.latex(y_expr)
                    },
                    "parameter": str(param),
                    "t_range": list(t_range),
                    "plot_image": img_base64,
                    "type": "parametric_plot"
                }
                
                return result
            
            except Exception as e:
                logger.error(f"Error evaluating parametric functions: {str(e)}")
                return {"error": f"Error evaluating parametric functions: {str(e)}", "type": "error"}
        
        except Exception as e:
            logger.error(f"Error plotting parametric curve: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def plot_3d(self, expr_str: str, var1_str: Optional[str] = None, var2_str: Optional[str] = None, 
               x_range: Tuple[float, float] = (-5, 5), y_range: Tuple[float, float] = (-5, 5), 
               points: int = 50, title: Optional[str] = None) -> Dict[str, Any]:
        """Plot a 3D surface.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var1_str: First variable. If None, 'x' will be used.
            var2_str: Second variable. If None, 'y' will be used.
            x_range: Range of x values to plot.
            y_range: Range of y values to plot.
            points: Number of points to plot in each dimension.
            title: Title of the plot.
            
        Returns:
            Dictionary containing the original expression, plot data,
            and a base64-encoded image of the plot.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Determine the variables
            if var1_str:
                var1 = sp.Symbol(var1_str)
            else:
                var1 = sp.Symbol('x')
            
            if var2_str:
                var2 = sp.Symbol(var2_str)
            else:
                var2 = sp.Symbol('y')
            
            # Convert the expression to a numpy function
            f = sp.lambdify((var1, var2), expr, "numpy")
            
            # Generate x and y values
            x = np.linspace(x_range[0], x_range[1], points)
            y = np.linspace(y_range[0], y_range[1], points)
            X, Y = np.meshgrid(x, y)
            
            # Calculate z values
            try:
                Z = f(X, Y)
                
                # Create the plot
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Plot the surface
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                
                # Add a color bar
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                # Set labels and title
                ax.set_xlabel(str(var1))
                ax.set_ylabel(str(var2))
                ax.set_zlabel(f"{expr_str}")
                ax.set_title(title if title else f"3D Plot of {expr_str}")
                
                # Save the plot to a buffer
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Encode the image as base64
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                
                # Close the plot to free memory
                plt.close()
                
                # Create the result
                result = {
                    "original": {
                        "string": expr_str,
                        "latex": sp.latex(expr)
                    },
                    "variables": [str(var1), str(var2)],
                    "x_range": list(x_range),
                    "y_range": list(y_range),
                    "plot_image": img_base64,
                    "type": "3d_plot"
                }
                
                return result
            
            except Exception as e:
                logger.error(f"Error evaluating 3D function: {str(e)}")
                return {"error": f"Error evaluating 3D function: {str(e)}", "type": "error"}
        
        except Exception as e:
            logger.error(f"Error plotting 3D surface: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def latex_render(self, expr_str: str) -> Dict[str, Any]:
        """Render a mathematical expression in LaTeX.
        
        Args:
            expr_str: String representation of a mathematical expression.
            
        Returns:
            Dictionary containing the original expression and LaTeX representation.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Generate LaTeX
            latex = sp.latex(expr)
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str
                },
                "latex": latex,
                "type": "latex_rendering"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error rendering LaTeX: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def evaluate_expression(self, expr_str: str, var_values: Dict[str, Union[str, float]]) -> Dict[str, Any]:
        """Evaluate a mathematical expression with specific variable values.
        
        Args:
            expr_str: String representation of a mathematical expression.
            var_values: Dictionary mapping variable names to values.
            
        Returns:
            Dictionary containing the original expression, variable values,
            and the evaluated result.
        """
        try:
            # Parse the expression
            expr = self.parse_expression(expr_str)
            
            # Parse variable values
            subs_dict = {}
            for var_name, value in var_values.items():
                var = sp.Symbol(var_name)
                if isinstance(value, str):
                    value = self.parse_expression(value)
                subs_dict[var] = value
            
            # Evaluate the expression
            result_expr = expr.subs(subs_dict)
            
            # Try to convert to a numerical value if possible
            try:
                result_value = float(result_expr)
            except:
                result_value = None
            
            self.last_result = result_expr
            
            # Create the result
            result = {
                "original": {
                    "string": expr_str,
                    "latex": sp.latex(expr)
                },
                "variable_values": {
                    var_name: {"string": str(value), "latex": sp.latex(value)}
                    for var_name, value in var_values.items()
                },
                "result": {
                    "string": str(result_expr),
                    "latex": sp.latex(result_expr)
                },
                "type": "expression_evaluation"
            }
            
            # Add numerical value if available
            if result_value is not None:
                result["numerical_value"] = result_value
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating expression: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def verify_identity(self, left_str: str, right_str: str) -> Dict[str, Any]:
        """Verify if two expressions are identical.
        
        Args:
            left_str: String representation of the left-hand side expression.
            right_str: String representation of the right-hand side expression.
            
        Returns:
            Dictionary containing the original expressions, verification result,
            and LaTeX representations.
        """
        try:
            # Parse the expressions
            left_expr = self.parse_expression(left_str)
            right_expr = self.parse_expression(right_str)
            
            # Check if the expressions are identical
            diff = sp.simplify(left_expr - right_expr)
            is_identical = diff == 0
            
            # Create the result
            result = {
                "left": {
                    "string": left_str,
                    "latex": sp.latex(left_expr)
                },
                "right": {
                    "string": right_str,
                    "latex": sp.latex(right_expr)
                },
                "is_identical": is_identical,
                "difference": {
                    "string": str(diff),
                    "latex": sp.latex(diff)
                },
                "type": "identity_verification"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error verifying identity: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def solve_inequality(self, ineq_str: str, var_str: Optional[str] = None) -> Dict[str, Any]:
        """Solve a mathematical inequality.
        
        Args:
            ineq_str: String representation of a mathematical inequality.
            var_str: Variable to solve for. If None, the first symbol in the inequality will be used.
            
        Returns:
            Dictionary containing the original inequality, solutions,
            and LaTeX representations.
        """
        try:
            # Replace inequality symbols with SymPy-compatible ones
            ineq_str = ineq_str.replace('≤', '<=')
            ineq_str = ineq_str.replace('≥', '>=')
            
            # Parse the inequality
            if '<=' in ineq_str:
                left_str, right_str = ineq_str.split('<=', 1)
                left_expr = self.parse_expression(left_str)
                right_expr = self.parse_expression(right_str)
                ineq = sp.Le(left_expr, right_expr)
            elif '>=' in ineq_str:
                left_str, right_str = ineq_str.split('>=', 1)
                left_expr = self.parse_expression(left_str)
                right_expr = self.parse_expression(right_str)
                ineq = sp.Ge(left_expr, right_expr)
            elif '<' in ineq_str:
                left_str, right_str = ineq_str.split('<', 1)
                left_expr = self.parse_expression(left_str)
                right_expr = self.parse_expression(right_str)
                ineq = sp.Lt(left_expr, right_expr)
            elif '>' in ineq_str:
                left_str, right_str = ineq_str.split('>', 1)
                left_expr = self.parse_expression(left_str)
                right_expr = self.parse_expression(right_str)
                ineq = sp.Gt(left_expr, right_expr)
            else:
                raise ValueError("Inequality must contain <, >, <=, or >=")
            
            # Determine the variable to solve for
            if var_str:
                var = sp.Symbol(var_str)
            else:
                # Use the first symbol in the inequality
                symbols = list(ineq.free_symbols)
                if not symbols:
                    raise ValueError("No variables found in the inequality.")
                var = symbols[0]
            
            # Solve the inequality
            solution = sp.solve_univariate_inequality(ineq, var)
            self.last_result = solution
            
            # Create the result
            result = {
                "original": {
                    "string": ineq_str,
                    "latex": sp.latex(ineq)
                },
                "variable": str(var),
                "solution": {
                    "string": str(solution),
                    "latex": sp.latex(solution)
                },
                "type": "inequality_solving"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error solving inequality: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_statistics(self, data_str: str) -> Dict[str, Any]:
        """Calculate basic statistics for a dataset.
        
        Args:
            data_str: String representation of a dataset (comma-separated values).
            
        Returns:
            Dictionary containing the original data and calculated statistics.
        """
        try:
            # Parse the data
            data_str = data_str.strip()
            if data_str.startswith('[') and data_str.endswith(']'):
                data_str = data_str[1:-1]
            
            # Split by commas and convert to numbers
            data = [float(x.strip()) for x in data_str.split(',')]
            
            # Calculate statistics
            n = len(data)
            mean = sum(data) / n
            variance = sum((x - mean) ** 2 for x in data) / n
            std_dev = variance ** 0.5
            sorted_data = sorted(data)
            median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
            
            # Calculate mode (most common value)
            from collections import Counter
            counter = Counter(data)
            mode_value, mode_count = counter.most_common(1)[0]
            
            # Calculate quartiles
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = sorted_data[q1_idx]
            q3 = sorted_data[q3_idx]
            iqr = q3 - q1
            
            # Calculate min and max
            min_val = min(data)
            max_val = max(data)
            
            # Create the result
            result = {
                "data": data,
                "count": n,
                "mean": mean,
                "median": median,
                "mode": {"value": mode_value, "count": mode_count},
                "variance": variance,
                "std_dev": std_dev,
                "min": min_val,
                "max": max_val,
                "range": max_val - min_val,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "type": "statistics_calculation"
            }
            
            return result
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {"error": str(e), "type": "error"}
    
    def calculate_probability(self, expr_str: str, dist_type: str, params: Dict[str, Union[str, float]]) -> Dict[str, Any]:
        """Calculate probability for a given distribution.
        
        Args:
            expr_str: String representation of a probability expression (e.g., "P(X <= 3)").
            dist_type: Type of distribution (normal, binomial, poisson, etc.).
            params: Parameters for the distribution.
            
        Returns:
            Dictionary containing the original expression, distribution parameters,
            and calculated probability.
        """
        try:
            # Parse the probability expression
            match = re.match(r"P\(([XY])\s*([<>=]+)\s*([\d.-]+)\)", expr_str)
            if not match:
                raise ValueError("Invalid probability expression format. Use P(X <= value) or similar.")
            
            var, op, value = match.groups()
            value = float(value)
            
            # Calculate the probability based on the distribution type
            if dist_type.lower() == "normal":
                # Parameters: mean, std_dev
                mean = float(params.get("mean", 0))
                std_dev = float(params.get("std_dev", 1))
                
                # Calculate the z-score
                z = (value - mean) / std_dev
                
                # Calculate the probability
                from scipy import stats
                if op == "<=":
                    prob = stats.norm.cdf(z)
                elif op == "<":
                    prob = stats.norm.cdf(z)
                elif op == ">=":
                    prob = 1 - stats.norm.cdf(z)
                elif op == ">":
                    prob = 1 - stats.norm.cdf(z)
                elif op == "==":
                    prob = 0  # Probability of a specific value in a continuous distribution is 0
                else:
                    raise ValueError(f"Unsupported operator: {op}")
                
                # Create the result
                result = {
                    "expression": expr_str,
                    "distribution": "normal",
                    "parameters": {"mean": mean, "std_dev": std_dev},
                    "probability": prob,
                    "type": "probability_calculation"
                }
                
                return result
            
            elif dist_type.lower() == "binomial":
                # Parameters: n, p
                n = int(params.get("n", 10))
                p = float(params.get("p", 0.5))
                
                # Calculate the probability
                from scipy import stats
                if op == "<=":
                    prob = stats.binom.cdf(value, n, p)
                elif op == "<":
                    prob = stats.binom.cdf(value - 1, n, p)
                elif op == ">=":
                    prob = 1 - stats.binom.cdf(value - 1, n, p)
                elif op == ">":
                    prob = 1 - stats.binom.cdf(value, n, p)
                elif op == "==":
                    prob = stats.binom.pmf(value, n, p)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
                
                # Create the result
                result = {
                    "expression": expr_str,
                    "distribution": "binomial",
                    "parameters": {"n": n, "p": p},
                    "probability": prob,
                    "type": "probability_calculation"
                }
                
                return result
            
            elif dist_type.lower() == "poisson":
                # Parameters: lambda
                lambda_val = float(params.get("lambda", 1))
                
                # Calculate the probability
                from scipy import stats
                if op == "<=":
                    prob = stats.poisson.cdf(value, lambda_val)
                elif op == "<":
                    prob = stats.poisson.cdf(value - 1, lambda_val)
                elif op == ">=":
                    prob = 1 - stats.poisson.cdf(value - 1, lambda_val)
                elif op == ">":
                    prob = 1 - stats.poisson.cdf(value, lambda_val)
                elif op == "==":
                    prob = stats.poisson.pmf(value, lambda_val)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
                
                # Create the result
                result = {
                    "expression": expr_str,
                    "distribution": "poisson",
                    "parameters": {"lambda": lambda_val},
                    "probability": prob,
                    "type": "probability_calculation"
                }
                
                return result
            
            else:
                raise ValueError(f"Unsupported distribution type: {dist_type}")
        
        except Exception as e:
            logger.error(f"Error calculating probability: {str(e)}")
            return {"error": str(e), "type": "error"}