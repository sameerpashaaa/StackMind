#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Planning System Module

This module implements the multi-step planning system for the AI Problem Solver,
allowing it to create, visualize, and refine solution plans.
"""

import logging
import uuid
from typing import Dict, List, Any, Optional, Union

from langchain.llms import BaseLLM
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate

from utils.helpers import get_current_datetime

logger = logging.getLogger(__name__)

class PlanningSystem:
    """
    Planning system for creating and managing solution plans.
    
    This system creates structured solution plans with steps, dependencies,
    and visualization capabilities.
    
    Attributes:
        llm: Language model for generating plans
        settings: Application settings
    """
    
    def __init__(self, llm: BaseLLM, settings):
        """
        Initialize the Planning System.
        
        Args:
            llm: Language model for generating plans
            settings: Application settings object
        """
        self.llm = llm
        self.settings = settings
        logger.info("Planning system initialized")
    
    def create_plan(self, problem: Dict[str, Any], domain: str) -> Dict[str, Any]:
        """
        Create a solution plan for a problem.
        
        Args:
            problem: The problem to create a plan for
            domain: The problem domain
            
        Returns:
            Dict[str, Any]: The solution plan
        """
        # Extract problem content
        problem_content = problem.get("content", "")
        
        # Create prompt for plan generation
        template = """
        I need a detailed solution plan for this problem in the {domain} domain:
        
        Problem: {problem}
        
        Please create a comprehensive solution plan that:
        1. Breaks down the problem into clear, manageable steps
        2. Specifies the order and dependencies between steps
        3. Identifies any required resources, tools, or information for each step
        4. Includes validation or verification points
        5. Anticipates potential challenges and includes contingency steps
        
        Format the plan as a structured JSON-like representation with:
        - A list of steps, each with an ID, description, and dependencies
        - Any branching or conditional logic clearly indicated
        - Estimated complexity or difficulty for each step
        
        The plan should be detailed enough to guide the solution process step by step.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem", "domain"]
        )
        
        # Generate plan using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that creates detailed solution plans for complex problems."),
            HumanMessage(content=prompt.format(
                problem=problem_content,
                domain=domain
            ))
        ]
        
        response = self.llm.generate([messages])
        plan_text = response.generations[0][0].text.strip()
        
        # Parse the plan into a structured format
        plan = self._parse_plan(plan_text)
        
        # Add metadata
        plan["id"] = str(uuid.uuid4())
        plan["problem_id"] = problem.get("id")
        plan["domain"] = domain
        plan["timestamp"] = get_current_datetime()
        plan["status"] = "created"
        
        logger.debug(f"Created plan with {len(plan.get('steps', []))} steps for problem: {problem.get('id')}")
        return plan
    
    def _parse_plan(self, plan_text: str) -> Dict[str, Any]:
        """
        Parse a plan text into a structured format.
        
        Args:
            plan_text: Raw plan text from LLM
            
        Returns:
            Dict[str, Any]: Structured plan
        """
        # Initialize plan structure
        plan = {
            "steps": [],
            "branches": [],
            "validations": [],
            "raw_text": plan_text
        }
        
        try:
            # Try to extract steps using a simple parsing approach
            lines = plan_text.split("\n")
            current_step = None
            step_number = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line starts a new step
                if line.startswith(f"Step {step_number}:") or line.startswith(f"Step {step_number}.") or \
                   line.startswith(f"{step_number}.") or line.startswith(f"{step_number}:") or \
                   line.lower().startswith(f"step {step_number}:") or line.lower().startswith(f"step {step_number}."):
                    
                    # Save the previous step if it exists
                    if current_step:
                        plan["steps"].append(current_step)
                    
                    # Start a new step
                    current_step = {
                        "id": f"step_{step_number}",
                        "step_number": step_number,
                        "description": line,
                        "details": "",
                        "dependencies": [],
                        "resources": [],
                        "complexity": "medium"
                    }
                    step_number += 1
                elif current_step:
                    # Check for specific step attributes
                    lower_line = line.lower()
                    if "dependencies:" in lower_line or "depends on:" in lower_line:
                        # Extract dependencies
                        deps_part = line.split(":", 1)[1].strip()
                        deps = [d.strip() for d in deps_part.split(",")]
                        current_step["dependencies"] = deps
                    elif "resources:" in lower_line or "tools:" in lower_line or "requires:" in lower_line:
                        # Extract resources
                        res_part = line.split(":", 1)[1].strip()
                        resources = [r.strip() for r in res_part.split(",")]
                        current_step["resources"] = resources
                    elif "complexity:" in lower_line or "difficulty:" in lower_line:
                        # Extract complexity
                        complexity_part = line.split(":", 1)[1].strip().lower()
                        if "high" in complexity_part or "difficult" in complexity_part:
                            current_step["complexity"] = "high"
                        elif "low" in complexity_part or "easy" in complexity_part:
                            current_step["complexity"] = "low"
                        else:
                            current_step["complexity"] = "medium"
                    else:
                        # Add to step details
                        current_step["details"] += line + "\n"
            
            # Add the last step
            if current_step:
                plan["steps"].append(current_step)
            
            # Extract branches and validations if present
            if "branch" in plan_text.lower() or "condition" in plan_text.lower():
                # Simple extraction of branches (this could be improved)
                branch_lines = [line for line in lines if "branch" in line.lower() or "condition" in line.lower() or "if" in line.lower()]
                for i, branch_line in enumerate(branch_lines):
                    plan["branches"].append({
                        "id": f"branch_{i+1}",
                        "description": branch_line.strip(),
                        "condition": branch_line.strip()
                    })
            
            if "validation" in plan_text.lower() or "verify" in plan_text.lower() or "check" in plan_text.lower():
                # Simple extraction of validations (this could be improved)
                validation_lines = [line for line in lines if "validation" in line.lower() or "verify" in line.lower() or "check" in line.lower()]
                for i, validation_line in enumerate(validation_lines):
                    plan["validations"].append({
                        "id": f"validation_{i+1}",
                        "description": validation_line.strip(),
                        "criteria": validation_line.strip()
                    })
        
        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            # If parsing fails, just store the raw text
            plan["parse_error"] = str(e)
        
        return plan
    
    def refine_plan(self, plan: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine a solution plan based on feedback.
        
        Args:
            plan: The original plan to refine
            feedback: User feedback
            
        Returns:
            Dict[str, Any]: The refined plan
        """
        # Extract plan and feedback content
        plan_text = plan.get("raw_text", "")
        feedback_content = feedback.get("content", "")
        
        # Create prompt for plan refinement
        template = """
        I need to refine this solution plan based on user feedback:
        
        Original Plan:
        {plan}
        
        User Feedback:
        {feedback}
        
        Please create a refined solution plan that addresses the user's feedback while maintaining the structure and comprehensiveness of the original plan.
        
        The refined plan should:
        1. Incorporate the user's suggestions or address their concerns
        2. Maintain or improve the clarity and organization of the original plan
        3. Ensure all steps are still logically connected and comprehensive
        4. Add any new steps or details needed based on the feedback
        5. Remove or modify any steps that were criticized or unnecessary
        
        Format the refined plan in the same structured way as the original plan.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["plan", "feedback"]
        )
        
        # Generate refined plan using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that refines solution plans based on user feedback."),
            HumanMessage(content=prompt.format(
                plan=plan_text,
                feedback=feedback_content
            ))
        ]
        
        response = self.llm.generate([messages])
        refined_plan_text = response.generations[0][0].text.strip()
        
        # Parse the refined plan
        refined_plan = self._parse_plan(refined_plan_text)
        
        # Add metadata
        refined_plan["id"] = str(uuid.uuid4())
        refined_plan["problem_id"] = plan.get("problem_id")
        refined_plan["domain"] = plan.get("domain")
        refined_plan["timestamp"] = get_current_datetime()
        refined_plan["status"] = "refined"
        refined_plan["original_plan_id"] = plan.get("id")
        refined_plan["feedback_id"] = feedback.get("id")
        
        logger.debug(f"Refined plan with {len(refined_plan.get('steps', []))} steps based on feedback")
        return refined_plan
    
    def create_alternative_plan(self, problem: Dict[str, Any], domain: str, original_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an alternative solution plan for a problem.
        
        Args:
            problem: The problem to create a plan for
            domain: The problem domain
            original_plan: The original plan to create an alternative to
            
        Returns:
            Dict[str, Any]: The alternative solution plan
        """
        # Extract problem and original plan content
        problem_content = problem.get("content", "")
        original_plan_text = original_plan.get("raw_text", "")
        
        # Create prompt for alternative plan generation
        template = """
        I need an alternative solution plan for this problem in the {domain} domain:
        
        Problem: {problem}
        
        Original Plan:
        {original_plan}
        
        Please create a comprehensive alternative solution plan that:
        1. Takes a significantly different approach from the original plan
        2. Breaks down the problem into clear, manageable steps
        3. Specifies the order and dependencies between steps
        4. Identifies any required resources, tools, or information for each step
        5. Includes validation or verification points
        
        The alternative plan should solve the same problem but use different methods, techniques, or perspectives.
        It should be detailed enough to guide the solution process step by step and formatted in the same structured way as the original plan.
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["problem", "domain", "original_plan"]
        )
        
        # Generate alternative plan using the LLM
        messages = [
            SystemMessage(content="You are an AI assistant that creates alternative solution plans for complex problems."),
            HumanMessage(content=prompt.format(
                problem=problem_content,
                domain=domain,
                original_plan=original_plan_text
            ))
        ]
        
        response = self.llm.generate([messages])
        alternative_plan_text = response.generations[0][0].text.strip()
        
        # Parse the alternative plan
        alternative_plan = self._parse_plan(alternative_plan_text)
        
        # Add metadata
        alternative_plan["id"] = str(uuid.uuid4())
        alternative_plan["problem_id"] = problem.get("id")
        alternative_plan["domain"] = domain
        alternative_plan["timestamp"] = get_current_datetime()
        alternative_plan["status"] = "alternative"
        alternative_plan["original_plan_id"] = original_plan.get("id")
        
        logger.debug(f"Created alternative plan with {len(alternative_plan.get('steps', []))} steps")
        return alternative_plan
    
    def get_plan_visualization(self, plan: Dict[str, Any], format: str = "tree") -> Dict[str, Any]:
        """
        Generate a visualization of a solution plan.
        
        Args:
            plan: The plan to visualize
            format: Visualization format (tree, graph, flowchart)
            
        Returns:
            Dict[str, Any]: Visualization data
        """
        # Extract plan steps and structure
        steps = plan.get("steps", [])
        branches = plan.get("branches", [])
        validations = plan.get("validations", [])
        
        # Create visualization based on format
        visualization = {
            "id": str(uuid.uuid4()),
            "plan_id": plan.get("id"),
            "format": format,
            "timestamp": get_current_datetime()
        }
        
        if format == "tree":
            # Create tree visualization
            tree_data = {
                "name": "Solution Plan",
                "children": []
            }
            
            # Add steps to the tree
            for step in steps:
                step_node = {
                    "name": f"Step {step.get('step_number')}: {step.get('description', '')[:50]}...",
                    "attributes": {
                        "complexity": step.get("complexity", "medium"),
                        "details": step.get("details", "")[:100] + "..."
                    },
                    "children": []
                }
                
                # Add resources as children
                for resource in step.get("resources", []):
                    step_node["children"].append({
                        "name": f"Resource: {resource}"
                    })
                
                tree_data["children"].append(step_node)
            
            # Add branches and validations
            if branches:
                branches_node = {
                    "name": "Branches",
                    "children": []
                }
                
                for branch in branches:
                    branches_node["children"].append({
                        "name": branch.get("description", "")[:50] + "..."
                    })
                
                tree_data["children"].append(branches_node)
            
            if validations:
                validations_node = {
                    "name": "Validations",
                    "children": []
                }
                
                for validation in validations:
                    validations_node["children"].append({
                        "name": validation.get("description", "")[:50] + "..."
                    })
                
                tree_data["children"].append(validations_node)
            
            visualization["data"] = tree_data
        
        elif format == "graph":
            # Create graph visualization
            graph_data = {
                "nodes": [],
                "edges": []
            }
            
            # Add steps as nodes
            for step in steps:
                graph_data["nodes"].append({
                    "id": step.get("id"),
                    "label": f"Step {step.get('step_number')}",
                    "title": step.get("description", ""),
                    "group": "step"
                })
                
                # Add edges based on dependencies
                for dep in step.get("dependencies", []):
                    if dep.startswith("step_"):
                        graph_data["edges"].append({
                            "from": dep,
                            "to": step.get("id"),
                            "arrows": "to"
                        })
            
            # Add branches and validations as nodes
            for branch in branches:
                graph_data["nodes"].append({
                    "id": branch.get("id"),
                    "label": "Branch",
                    "title": branch.get("description", ""),
                    "group": "branch"
                })
            
            for validation in validations:
                graph_data["nodes"].append({
                    "id": validation.get("id"),
                    "label": "Validation",
                    "title": validation.get("description", ""),
                    "group": "validation"
                })
            
            visualization["data"] = graph_data
        
        elif format == "flowchart":
            # Create flowchart visualization (simplified mermaid syntax)
            flowchart_data = "flowchart TD\n"
            
            # Add steps
            for step in steps:
                step_id = step.get("id").replace("_", "")
                flowchart_data += f"  {step_id}[\"Step {step.get('step_number')}: {step.get('description', '')[:30]}...\"]\n"
            
            # Add connections based on dependencies
            for step in steps:
                step_id = step.get("id").replace("_", "")
                for dep in step.get("dependencies", []):
                    if dep.startswith("step_"):
                        dep_id = dep.replace("_", "")
                        flowchart_data += f"  {dep_id} --> {step_id}\n"
            
            # Add branches
            for i, branch in enumerate(branches):
                branch_id = f"branch{i+1}"
                flowchart_data += f"  {branch_id}{{\"Branch: {branch.get('description', '')[:30]}...\"}}\n"
            
            # Add validations
            for i, validation in enumerate(validations):
                validation_id = f"validation{i+1}"
                flowchart_data += f"  {validation_id}[/\"Validation: {validation.get('description', '')[:30]}...\"/]\n"
            
            visualization["data"] = flowchart_data
        
        else:
            # Unsupported format
            visualization["error"] = f"Unsupported visualization format: {format}"
        
        logger.debug(f"Generated {format} visualization for plan: {plan.get('id')}")
        return visualization
    
    def get_plan_progress(self, plan: Dict[str, Any], completed_steps: List[str]) -> Dict[str, Any]:
        """
        Calculate and visualize progress on a solution plan.
        
        Args:
            plan: The solution plan
            completed_steps: List of completed step IDs
            
        Returns:
            Dict[str, Any]: Progress information
        """
        steps = plan.get("steps", [])
        total_steps = len(steps)
        completed_count = len(completed_steps)
        
        # Calculate progress percentage
        progress_percentage = (completed_count / total_steps * 100) if total_steps > 0 else 0
        
        # Determine current step
        current_step = None
        next_steps = []
        
        for step in steps:
            step_id = step.get("id")
            if step_id not in completed_steps:
                # Check if dependencies are satisfied
                dependencies = step.get("dependencies", [])
                dependencies_satisfied = True
                
                for dep in dependencies:
                    if dep not in completed_steps:
                        dependencies_satisfied = False
                        break
                
                if dependencies_satisfied:
                    if current_step is None:
                        current_step = step
                    else:
                        next_steps.append(step)
        
        # Create progress object
        progress = {
            "id": str(uuid.uuid4()),
            "plan_id": plan.get("id"),
            "timestamp": get_current_datetime(),
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "progress_percentage": progress_percentage,
            "current_step": current_step,
            "next_steps": next_steps,
            "completed_step_ids": completed_steps
        }
        
        logger.debug(f"Calculated plan progress: {progress_percentage:.1f}% complete")
        return progress