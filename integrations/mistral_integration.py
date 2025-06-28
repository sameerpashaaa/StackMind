#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mistral API integration for StackMind"""
import os
import logging
from typing import List, Dict, Any

from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)

def get_mistral_model(model_name: str = "mistral-large-latest", temperature: float = 0.7, max_tokens: int = 2000) -> ChatMistralAI:
    """Initialize Mistral model with API key from environment"""
    try:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            logger.warning("MISTRAL_API_KEY not found in environment variables")
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        model = ChatMistralAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            mistral_api_key=api_key
        )
        logger.info(f"Initialized Mistral model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize Mistral model: {e}")
        raise

def generate_completion(prompt: str, model_name: str = "mistral-large-latest", temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Generate text completion from prompt"""
    try:
        model = get_mistral_model(model_name, temperature, max_tokens)
        messages = [HumanMessage(content=prompt)]
        response = model.generate([messages])
        generated_text = response.generations[0][0].text
        return generated_text
    except Exception as e:
        logger.error(f"Failed to generate completion: {e}")
        return ""

def generate_chat_response(messages: List[Dict[str, str]], model_name: str = "mistral-large-latest", temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """Generate chat response from message history"""
    try:
        model = get_mistral_model(model_name, temperature, max_tokens)
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
        generated_text = response.generations[0][0].text
        return generated_text
    except Exception as e:
        logger.error(f"Failed to generate chat response: {e}")
        return ""