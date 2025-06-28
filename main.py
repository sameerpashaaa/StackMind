#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import logging
from dotenv import load_dotenv

# Import core modules
from core.agent import ProblemSolverAgent
from interfaces.cli import CommandLineInterface
from interfaces.api import start_api_server
from config.settings import Settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("problem_solver.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='StackMind')
    parser.add_argument('--api', action='store_true', help='Start the API server')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--config', type=str, default='default', help='Configuration profile to use')
    return parser.parse_args()

def main():
    """Main function to initialize and run StackMind"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set debug mode if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Load settings
    settings = Settings(profile=args.config)
    logger.info(f"Loaded configuration profile: {args.config}")
    
    # Check for API keys based on the selected LLM provider
    if settings.llm_provider == "mistral":
        if not os.getenv("MISTRAL_API_KEY"):
            logger.warning("MISTRAL_API_KEY not found in environment variables. Please set it in .env file.")
            print("Warning: MISTRAL_API_KEY not found. Please set it in .env file or export it as an environment variable.")
    elif settings.llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables. Please set it in .env file.")
            print("Warning: OPENAI_API_KEY not found. Please set it in .env file or export it as an environment variable.")
    else:
        logger.warning(f"No API key check implemented for LLM provider: {settings.llm_provider}")
    
    # Initialize the agent
    agent = ProblemSolverAgent(settings=settings)
    logger.info("Problem Solver Agent initialized")
    
    # Start the appropriate interface
    if args.api:
        logger.info("Starting API server")
        start_api_server(agent, settings)
    else:
        logger.info("Starting command-line interface")
        cli = CommandLineInterface(agent, settings)
        cli.start()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
        print("\nApplication terminated by user. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred: {e}")