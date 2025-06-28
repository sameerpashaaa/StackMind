#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Problem Solver Module

This module implements a general-purpose problem solver that can handle
a wide range of problems across different domains.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from langchain.llms import BaseLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate

from utils.helpers import get_current_datetime

logger = logging.getLogger(__name__)

class GeneralSolver:
    """
    General-purpose problem solver for handling a wide range of problems.
    
    This solver uses a flexible approach that can be applied to various
    problem types when a more specialized domain solver is not available.
    
    Attributes:
        llm: Language model for generating solutions
        settings: Application settings
    """
    
    def __init__(self, llm: BaseLLM, settings):
        """
        Initialize the General Solver.
        
        Args:
            llm: Language model for generating solutions
            settings: Application settings object
        """
        self.llm = llm
        self.settings = settings
        logger.info("General solver initialized")
    
    def solve(self, problem: Dict[str, Any], plan: Optional[Dict[str, Any]] = None, 
             feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a general problem using the provided plan.
        
        Args:
            problem: The problem to solve
            plan: Optional solution plan
            feedback: Optional user feedback
            
        Returns:
            Dict[str, Any]: The solution
        """
        # Extract problem content
        problem_content = problem.get("content", "")
        
        # Extract plan if provided
        plan_text = ""
        if plan:
            plan_text = plan.get("raw_text", "")
        
        # Extract feedback if provided
        feedback_text = ""
        if feedback:
            feedback_text = feedback.get("content", "")
        
        # Create prompt for solution generation
        template = self._create_solution_prompt(bool(plan), bool(feedback))
        
        # Generate solution using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that solves problems using a systematic approach."),
            HumanMessage(content=template.format(
                problem=problem_content,
                plan=plan_text,
                feedback=feedback_text
            ))
        ]
        
        response = self.llm.generate([messages])
        solution_text = response.generations[0][0].text.strip()
        
        # Create solution object
        solution = {
            "id": str(uuid.uuid4()),
            "problem_id": problem.get("id"),
            "content": solution_text,
            "timestamp": get_current_datetime(),
            "domain": "general"
        }
        
        # Add plan and feedback references if provided
        if plan:
            solution["plan_id"] = plan.get("id")
        
        if feedback:
            solution["feedback_id"] = feedback.get("id")
            solution["is_refinement"] = True
        
        logger.debug(f"Generated solution for problem: {problem.get('id')}")
        return solution
    
    def _create_solution_prompt(self, has_plan: bool = False, has_feedback: bool = False) -> PromptTemplate:
        """
        Create a prompt template for generating solutions.
        
        Args:
            has_plan: Whether a plan is provided
            has_feedback: Whether user feedback is provided
            
        Returns:
            PromptTemplate: The prompt template
        """
        if has_plan and has_feedback:
            template = """
            I need a solution to this problem, following the provided plan and incorporating user feedback:
            
            Problem: {problem}
            
            Solution Plan:
            {plan}
            
            User Feedback:
            {feedback}
            
            Please provide a comprehensive solution that:
            1. Follows the steps outlined in the plan
            2. Addresses the user's feedback and concerns
            3. Explains the reasoning behind each part of the solution
            4. Includes any necessary calculations, algorithms, or procedures
            5. Verifies the solution against the original problem requirements
            
            The solution should be detailed, clear, and directly applicable to the problem.
            """
        elif has_plan:
            template = """
            I need a solution to this problem, following the provided plan:
            
            Problem: {problem}
            
            Solution Plan:
            {plan}
            
            Please provide a comprehensive solution that:
            1. Follows the steps outlined in the plan
            2. Explains the reasoning behind each part of the solution
            3. Includes any necessary calculations, algorithms, or procedures
            4. Verifies the solution against the original problem requirements
            
            The solution should be detailed, clear, and directly applicable to the problem.
            """
        elif has_feedback:
            template = """
            I need a refined solution to this problem, incorporating user feedback:
            
            Problem: {problem}
            
            User Feedback:
            {feedback}
            
            Please provide a comprehensive solution that:
            1. Addresses the user's feedback and concerns
            2. Takes a systematic approach to solving the problem
            3. Explains the reasoning behind each part of the solution
            4. Includes any necessary calculations, algorithms, or procedures
            5. Verifies the solution against the original problem requirements
            
            The solution should be detailed, clear, and directly applicable to the problem.
            """
        else:
            template = """
            I need a solution to this problem:
            
            Problem: {problem}
            
            Please provide a comprehensive solution that:
            1. Takes a systematic approach to solving the problem
            2. Explains the reasoning behind each part of the solution
            3. Includes any necessary calculations, algorithms, or procedures
            4. Verifies the solution against the problem requirements
            
            The solution should be detailed, clear, and directly applicable to the problem.
            """
        
        return PromptTemplate(template=template, input_variables=["problem", "plan", "feedback"])
    
    def verify(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify a solution for correctness and completeness.
        
        Args:
            solution: The solution to verify
            
        Returns:
            Dict[str, Any]: Verification results
        """
        # Get the problem from the solution
        problem_id = solution.get("problem_id")
        if not problem_id:
            logger.warning("Cannot verify solution without problem_id")
            return {
                "is_correct": False,
                "error": "Missing problem_id in solution"
            }
        
        # Extract solution content
        solution_content = solution.get("content", "")
        
        # Create prompt for verification
        template = """
        I need to verify this solution for correctness and completeness:
        
        Solution: {solution}
        
        Please verify the solution by:
        1. Checking if it correctly addresses the problem
        2. Verifying any calculations, algorithms, or procedures
        3. Identifying any errors, omissions, or inconsistencies
        4. Assessing the clarity and completeness of the explanation
        5. Determining if the solution is optimal or if there are better approaches
        
        Provide a detailed verification report with specific feedback on each aspect.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["solution"]
        )
        
        # Generate verification using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that verifies solutions for correctness and completeness."),
            HumanMessage(content=prompt.format(solution=solution_content))
        ]
        
        response = self.llm.generate([messages])
        verification_text = response.generations[0][0].text.strip()
        
        # Determine if the solution is correct based on the verification
        is_correct = True
        if "incorrect" in verification_text.lower() or "error" in verification_text.lower() or \
           "wrong" in verification_text.lower() or "mistake" in verification_text.lower() or \
           "issue" in verification_text.lower() or "problem" in verification_text.lower():
            is_correct = False
        
        # Create verification object
        verification = {
            "id": str(uuid.uuid4()),
            "solution_id": solution.get("id"),
            "content": verification_text,
            "is_correct": is_correct,
            "timestamp": get_current_datetime()
        }
        
        logger.debug(f"Verified solution: {solution.get('id')}, Result: {is_correct}")
        return verification
    
    def explain(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a solution.
        
        Args:
            solution: The solution to explain
            
        Returns:
            Dict[str, Any]: Detailed explanation
        """
        # Extract solution content
        solution_content = solution.get("content", "")
        
        # Create prompt for explanation
        template = """
        I need a detailed explanation of this solution:
        
        Solution: {solution}
        
        Please provide a comprehensive explanation that:
        1. Breaks down the solution into clear, understandable parts
        2. Explains the reasoning and methodology behind each part
        3. Clarifies any complex concepts, calculations, or procedures
        4. Relates the solution to general principles or concepts
        5. Discusses any alternative approaches that could have been used
        
        The explanation should be educational and accessible, helping someone understand not just what the solution is, but why it works and how it was derived.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["solution"]
        )
        
        # Generate explanation using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that provides detailed explanations of solutions."),
            HumanMessage(content=prompt.format(solution=solution_content))
        ]
        
        response = self.llm.generate([messages])
        explanation_text = response.generations[0][0].text.strip()
        
        # Create explanation object
        explanation = {
            "id": str(uuid.uuid4()),
            "solution_id": solution.get("id"),
            "content": explanation_text,
            "timestamp": get_current_datetime()
        }
        
        logger.debug(f"Generated explanation for solution: {solution.get('id')}")
        return explanation