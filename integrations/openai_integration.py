#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI Integration Module

This module provides functions for integrating with OpenAI's API services,
including language models, embeddings, and other AI capabilities.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union

from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

def get_openai_model(model_name: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 2000) -> ChatOpenAI:
    """
    Initialize and return an OpenAI language model.
    
    Args:
        model_name (str): Name of the OpenAI model to use
        temperature (float): Temperature parameter for controlling randomness
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        ChatOpenAI: Initialized OpenAI language model
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        model = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=api_key
        )
        
        logger.info(f"Initialized OpenAI model: {model_name}")
        return model
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI model: {e}")
        raise

def get_openai_embeddings() -> OpenAIEmbeddings:
    """
    Initialize and return OpenAI embeddings.
    
    Returns:
        OpenAIEmbeddings: Initialized OpenAI embeddings
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        
        logger.info("Initialized OpenAI embeddings")
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embeddings: {e}")
        raise

def generate_completion(prompt: str, model_name: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Generate a completion using OpenAI's API.
    
    Args:
        prompt (str): Prompt text
        model_name (str): Name of the OpenAI model to use
        temperature (float): Temperature parameter for controlling randomness
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated completion text
    """
    try:
        model = get_openai_model(model_name, temperature, max_tokens)
        messages = [HumanMessage(content=prompt)]
        response = model.generate([messages])
        
        # Extract the generated text from the response
        generated_text = response.generations[0][0].text
        
        return generated_text
    
    except Exception as e:
        logger.error(f"Failed to generate completion: {e}")
        return ""

def generate_chat_response(messages: List[Dict[str, str]], model_name: str = "gpt-4", temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Generate a chat response using OpenAI's API.
    
    Args:
        messages (List[Dict[str, str]]): List of message dictionaries with 'role' and 'content' keys
        model_name (str): Name of the OpenAI model to use
        temperature (float): Temperature parameter for controlling randomness
        max_tokens (int): Maximum number of tokens to generate
        
    Returns:
        str: Generated chat response text
    """
    try:
        model = get_openai_model(model_name, temperature, max_tokens)
        
        # Convert message dictionaries to LangChain message objects
        langchain_messages = []
        for message in messages:
            role = message.get('role', '').lower()
            content = message.get('content', '')
            
            if role == 'system':
                langchain_messages.append(SystemMessage(content=content))
            elif role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                langchain_messages.append(AIMessage(content=content))
        
        response = model.generate([langchain_messages])
        
        # Extract the generated text from the response
        generated_text = response.generations[0][0].text
        
        return generated_text
    
    except Exception as e:
        logger.error(f"Failed to generate chat response: {e}")
        return ""