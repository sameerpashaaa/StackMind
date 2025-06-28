#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reasoning Engine Module

This module implements the chain-of-thought reasoning engine for the AI Problem Solver,
allowing it to break down complex problems and provide transparent step-by-step reasoning.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from langchain.llms import BaseLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from utils.helpers import get_current_datetime

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Reasoning engine for step-by-step problem solving.
    
    This engine implements chain-of-thought reasoning to break down complex problems
    into smaller, traceable sub-problems and provide transparent reasoning steps.
    
    Attributes:
        llm: Language model for generating reasoning
        settings: Application settings
    """
    
    def __init__(self, llm: BaseLLM, settings):
        """
        Initialize the Reasoning Engine.
        
        Args:
            llm: Language model for generating reasoning
            settings: Application settings object
        """
        self.llm = llm
        self.settings = settings
        logger.info("Reasoning engine initialized")
    
    def get_reasoning_chain(self, problem: Dict[str, Any], solution: Dict[str, Any], 
                           feedback: Optional[Dict[str, Any]] = None,
                           is_alternative: bool = False) -> List[Dict[str, Any]]:
        """
        Generate a chain of reasoning steps for a problem and solution.
        
        Args:
            problem: The problem input
            solution: The solution output
            feedback: Optional user feedback
            is_alternative: Whether this is an alternative solution
            
        Returns:
            List[Dict[str, Any]]: List of reasoning steps
        """
        # Extract problem content
        problem_content = problem.get("content", "")
        
        # Extract solution content
        solution_content = solution.get("content", "")
        
        # Extract feedback content if provided
        feedback_content = ""
        if feedback:
            feedback_content = feedback.get("content", "")
        
        # Create prompt for reasoning chain
        prompt_template = self._create_reasoning_prompt(is_alternative, bool(feedback))
        
        # Generate reasoning using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that provides detailed step-by-step reasoning for problem-solving."),
            HumanMessage(content=prompt_template.format(
                problem=problem_content,
                solution=solution_content,
                feedback=feedback_content
            ))
        ]
        
        response = self.llm.generate([messages])
        reasoning_text = response.generations[0][0].text.strip()
        
        # Parse the reasoning steps
        reasoning_steps = self._parse_reasoning_steps(reasoning_text)
        
        logger.debug(f"Generated {len(reasoning_steps)} reasoning steps")
        return reasoning_steps
    
    def _create_reasoning_prompt(self, is_alternative: bool = False, has_feedback: bool = False) -> PromptTemplate:
        """
        Create a prompt template for generating reasoning chains.
        
        Args:
            is_alternative: Whether this is an alternative solution
            has_feedback: Whether user feedback is provided
            
        Returns:
            PromptTemplate: The prompt template
        """
        if is_alternative:
            if has_feedback:
                template = """
                I need a detailed chain-of-thought reasoning for an alternative solution to this problem.
                
                Problem: {problem}
                
                User feedback on previous solution: {feedback}
                
                Alternative solution: {solution}
                
                Please provide a step-by-step breakdown of the reasoning process for this alternative solution.
                For each step, explain the thought process, any assumptions made, and how it addresses the user's feedback.
                
                Format your response as a numbered list of steps, with each step clearly explaining one part of the reasoning process.
                Each step should be self-contained and build upon previous steps.
                """
            else:
                template = """
                I need a detailed chain-of-thought reasoning for an alternative solution to this problem.
                
                Problem: {problem}
                
                Alternative solution: {solution}
                
                Please provide a step-by-step breakdown of the reasoning process for this alternative solution.
                For each step, explain the thought process, any assumptions made, and how it differs from a standard approach.
                
                Format your response as a numbered list of steps, with each step clearly explaining one part of the reasoning process.
                Each step should be self-contained and build upon previous steps.
                """
        else:
            if has_feedback:
                template = """
                I need a detailed chain-of-thought reasoning for a refined solution to this problem.
                
                Problem: {problem}
                
                User feedback: {feedback}
                
                Refined solution: {solution}
                
                Please provide a step-by-step breakdown of the reasoning process for this refined solution.
                For each step, explain the thought process, any assumptions made, and how it addresses the user's feedback.
                
                Format your response as a numbered list of steps, with each step clearly explaining one part of the reasoning process.
                Each step should be self-contained and build upon previous steps.
                """
            else:
                template = """
                I need a detailed chain-of-thought reasoning for this problem.
                
                Problem: {problem}
                
                Solution: {solution}
                
                Please provide a step-by-step breakdown of the reasoning process that leads to this solution.
                For each step, explain the thought process and any assumptions made.
                
                Format your response as a numbered list of steps, with each step clearly explaining one part of the reasoning process.
                Each step should be self-contained and build upon previous steps.
                """
        
        return PromptTemplate(template=template, input_variables=["problem", "solution", "feedback"])
    
    def _parse_reasoning_steps(self, reasoning_text: str) -> List[Dict[str, Any]]:
        """
        Parse reasoning text into structured steps.
        
        Args:
            reasoning_text: Raw reasoning text from LLM
            
        Returns:
            List[Dict[str, Any]]: Structured reasoning steps
        """
        steps = []
        current_step = ""
        step_number = 1
        
        # Split the text into lines
        lines = reasoning_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new step
            if line.startswith(f"{step_number}.") or line.startswith(f"Step {step_number}:") or \
               line.startswith(f"Step {step_number}.") or line.startswith(f"{step_number}:"):
                # Save the previous step if it exists
                if current_step:
                    steps.append({
                        "id": str(uuid.uuid4()),
                        "step_number": step_number - 1,
                        "content": current_step.strip(),
                        "timestamp": get_current_datetime()
                    })
                
                # Start a new step
                current_step = line
                step_number += 1
            else:
                # Continue the current step
                current_step += "\n" + line
        
        # Add the last step
        if current_step:
            steps.append({
                "id": str(uuid.uuid4()),
                "step_number": step_number - 1,
                "content": current_step.strip(),
                "timestamp": get_current_datetime()
            })
        
        # If no steps were found using the numbered format, try to split by paragraphs
        if not steps:
            paragraphs = reasoning_text.split("\n\n")
            for i, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    steps.append({
                        "id": str(uuid.uuid4()),
                        "step_number": i + 1,
                        "content": paragraph.strip(),
                        "timestamp": get_current_datetime()
                    })
        
        return steps
    
    def explain_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a solution.
        
        Args:
            solution: The solution to explain
            
        Returns:
            Dict[str, Any]: Detailed explanation
        """
        # Extract solution content and reasoning steps
        solution_content = solution.get("content", "")
        reasoning_steps = solution.get("reasoning_steps", [])
        
        # Create prompt for explanation
        template = """
        I need a detailed explanation of this solution:
        
        Solution: {solution}
        
        Reasoning steps:
        {reasoning_steps}
        
        Please provide a comprehensive explanation that:
        1. Summarizes the solution approach
        2. Explains the key insights and techniques used
        3. Clarifies any complex or non-obvious parts
        4. Discusses the strengths and potential limitations of this solution
        5. Relates the solution to the underlying principles or concepts
        
        Make the explanation educational and accessible, suitable for someone who wants to understand not just what the solution is, but why it works and how it was derived.
        """
        
        # Format reasoning steps as text
        reasoning_text = ""
        for step in reasoning_steps:
            reasoning_text += f"Step {step.get('step_number')}: {step.get('content')}\n\n"
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["solution", "reasoning_steps"]
        )
        
        # Generate explanation using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that provides detailed explanations of problem solutions."),
            HumanMessage(content=prompt.format(
                solution=solution_content,
                reasoning_steps=reasoning_text
            ))
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
    
    def analyze_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a problem to identify key components and potential approaches.
        
        Args:
            problem: The problem to analyze
            
        Returns:
            Dict[str, Any]: Problem analysis
        """
        # Extract problem content
        problem_content = problem.get("content", "")
        
        # Create prompt for problem analysis
        template = """
        I need a detailed analysis of this problem:
        
        Problem: {problem}
        
        Please provide a comprehensive analysis that:
        1. Identifies the key components and constraints of the problem
        2. Breaks down the problem into smaller sub-problems or steps
        3. Identifies potential approaches or strategies for solving the problem
        4. Highlights any potential challenges or edge cases
        5. Suggests relevant concepts, formulas, or techniques that might be useful
        
        The analysis should help understand the problem structure and guide the solution process.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem"]
        )
        
        # Generate analysis using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that analyzes problems to guide the solution process."),
            HumanMessage(content=prompt.format(problem=problem_content))
        ]
        
        response = self.llm.generate([messages])
        analysis_text = response.generations[0][0].text.strip()
        
        # Create analysis object
        analysis = {
            "id": str(uuid.uuid4()),
            "problem_id": problem.get("id"),
            "content": analysis_text,
            "timestamp": get_current_datetime()
        }
        
        logger.debug(f"Generated analysis for problem: {problem.get('id')}")
        return analysis
    
    def compare_solutions(self, solution1: Dict[str, Any], solution2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two solutions to the same problem.
        
        Args:
            solution1: First solution
            solution2: Second solution
            
        Returns:
            Dict[str, Any]: Comparison analysis
        """
        # Extract solution contents
        solution1_content = solution1.get("content", "")
        solution2_content = solution2.get("content", "")
        
        # Create prompt for solution comparison
        template = """
        I need a detailed comparison of these two solutions to the same problem:
        
        Solution 1: {solution1}
        
        Solution 2: {solution2}
        
        Please provide a comprehensive comparison that:
        1. Summarizes the key differences in approach between the two solutions
        2. Analyzes the strengths and weaknesses of each solution
        3. Compares the efficiency, elegance, and robustness of the solutions
        4. Identifies scenarios where one solution might be preferable over the other
        5. Suggests potential improvements or hybrid approaches
        
        The comparison should be balanced and objective, highlighting the merits of both solutions.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["solution1", "solution2"]
        )
        
        # Generate comparison using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that compares different solutions to the same problem."),
            HumanMessage(content=prompt.format(
                solution1=solution1_content,
                solution2=solution2_content
            ))
        ]
        
        response = self.llm.generate([messages])
        comparison_text = response.generations[0][0].text.strip()
        
        # Create comparison object
        comparison = {
            "id": str(uuid.uuid4()),
            "solution1_id": solution1.get("id"),
            "solution2_id": solution2.get("id"),
            "content": comparison_text,
            "timestamp": get_current_datetime()
        }
        
        logger.debug(f"Generated comparison between solutions: {solution1.get('id')} and {solution2.get('id')}")
        return comparison