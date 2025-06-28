#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
External Tools Integration Module

This module provides integration with external tools and services
that can be used by the AI Problem Solver for enhanced capabilities.
"""

import os
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union

from langchain.tools import BaseTool, Tool
from langchain.agents import Tool as LangChainTool

from core.code_execution import execute_python_code, execute_javascript_code, execute_shell_command
from utils.helpers import safe_json_loads, safe_json_dumps

logger = logging.getLogger(__name__)

def get_available_tools(settings) -> List[BaseTool]:
    """
    Get a list of available tools based on the application settings.
    
    Args:
        settings: Application settings object
        
    Returns:
        List[BaseTool]: List of available tools
    """
    tools = []
    
    # Add tools based on settings
    integrations_settings = settings.get_section("integrations")
    
    # Python code execution tool
    if integrations_settings.get("enable_code_execution", True):
        tools.append(
            LangChainTool(
                name="python_executor",
                func=execute_python_code,
                description="Execute Python code and return the result"
            )
        )
    
    # JavaScript code execution tool
    if integrations_settings.get("enable_code_execution", True):
        tools.append(
            LangChainTool(
                name="javascript_executor",
                func=execute_javascript_code,
                description="Execute JavaScript code using Node.js and return the result"
            )
        )
    
    # Shell command execution tool (with safety checks)
    if integrations_settings.get("enable_code_execution", True) and not integrations_settings.get("safe_mode", True):
        tools.append(
            LangChainTool(
                name="shell_executor",
                func=execute_shell_command,
                description="Execute shell commands (use with caution)"
            )
        )
    
    # Add more tools as needed
    
    return tools

def execute_tool(tool_name: str, tool_input: Any) -> Dict[str, Any]:
    """
    Execute a tool by name with the given input.
    
    Args:
        tool_name (str): Name of the tool to execute
        tool_input (Any): Input for the tool
        
    Returns:
        Dict[str, Any]: Tool execution result
    """
    try:
        # Get available tools
        tools = get_available_tools(None)  # Replace with actual settings
        
        # Find the requested tool
        tool = next((t for t in tools if t.name == tool_name), None)
        
        if tool is None:
            logger.warning(f"Tool not found: {tool_name}")
            return {"error": f"Tool not found: {tool_name}"}
        
        # Execute the tool
        result = tool.run(tool_input)
        
        return {"result": result}
    
    except Exception as e:
        logger.error(f"Failed to execute tool {tool_name}: {e}")
        return {"error": str(e)}

def check_tool_availability(tool_name: str) -> bool:
    """
    Check if a specific tool is available in the current environment.
    
    Args:
        tool_name (str): Name of the tool to check
        
    Returns:
        bool: True if the tool is available, False otherwise
    """
    try:
        if tool_name == "python_executor":
            # Check if Python is available
            subprocess.run(["python", "--version"], capture_output=True, check=True)
            return True
        
        elif tool_name == "javascript_executor":
            # Check if Node.js is available
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            return True
        
        elif tool_name == "shell_executor":
            # Shell is generally available, but we might want to check specific commands
            return True
        
        # Add more tool availability checks as needed
        
        return False
    
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning(f"Tool not available: {tool_name}")
        return False

def register_custom_tool(name: str, func: callable, description: str) -> BaseTool:
    """
    Register a custom tool for use with the AI Problem Solver.
    
    Args:
        name (str): Name of the tool
        func (callable): Function to execute when the tool is called
        description (str): Description of the tool
        
    Returns:
        BaseTool: Registered tool
    """
    try:
        tool = LangChainTool(
            name=name,
            func=func,
            description=description
        )
        
        logger.info(f"Registered custom tool: {name}")
        return tool
    
    except Exception as e:
        logger.error(f"Failed to register custom tool {name}: {e}")
        raise