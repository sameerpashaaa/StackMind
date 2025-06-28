#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple

from langchain.llms import BaseLLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

from core.memory import MemorySystem
from core.reasoning import ReasoningEngine
from core.planning import PlanningSystem
from processors.text_processor import TextProcessor
from processors.image_processor import ImageProcessor
from processors.voice_processor import VoiceProcessor
from processors.code_processor import CodeProcessor
from domains.general_solver import GeneralSolver
from integrations.openai_integration import get_openai_model
from utils.helpers import get_current_datetime
from integrations.mistral_integration import get_mistral_model

logger = logging.getLogger(__name__)

class ProblemSolverAgent:
    def __init__(self, settings):
        self.settings = settings
        self.session_id = str(uuid.uuid4())
        logger.info(f"Initializing Problem Solver Agent with session ID: {self.session_id}")
        
        self.llm = self._initialize_llm()
        
        self.memory = MemorySystem(settings)
        self.reasoning = ReasoningEngine(self.llm, settings)
        self.planning = PlanningSystem(self.llm, settings)
        
        self.processors = self._initialize_processors()
        
        self.domain_solvers = self._initialize_domain_solvers()
        
        self.current_problem = None
        self.current_plan = None
        self.current_step = 0
        self.solution_history = []
        
        logger.info("Problem Solver Agent initialized successfully")

    def _initialize_llm(self) -> BaseLLM:
        llm_settings = self.settings.get_section("llm")
        provider = llm_settings.get("provider", "openai")
        
        if provider == "openai":
            model_name = llm_settings.get("model", "gpt-4")
            temperature = llm_settings.get("temperature", 0.7)
            max_tokens = llm_settings.get("max_tokens", 2000)
            logger.info(f"Initializing OpenAI model: {model_name}")
            return get_openai_model(model_name, temperature, max_tokens)
        elif provider == "mistral":
            model_name = llm_settings.get("model", "mistral-large-latest")
            temperature = llm_settings.get("temperature", 0.7)
            max_tokens = llm_settings.get("max_tokens", 2000)
            logger.info(f"Initializing Mistral model: {model_name}")
            return get_mistral_model(model_name, temperature, max_tokens)
        else:
            logger.warning(f"Unsupported LLM provider: {provider}. Falling back to OpenAI.")
            return get_openai_model("gpt-4", 0.7, 2000)

    def _initialize_processors(self) -> Dict[str, Any]:
        processors = {}
        input_settings = self.settings.get_section("input_processing")
        
        if input_settings.get("enable_text", True):
            processors["text"] = TextProcessor()
        
        if input_settings.get("enable_image", True):
            processors["image"] = ImageProcessor(
                max_size=input_settings.get("max_image_size", 10 * 1024 * 1024),
                supported_formats=input_settings.get("supported_image_formats", ["jpg", "jpeg", "png"])
            )
        
        if input_settings.get("enable_voice", True):
            processors["voice"] = VoiceProcessor()
        
        if input_settings.get("enable_code", True):
            processors["code"] = CodeProcessor(
                supported_languages=input_settings.get("supported_code_languages", ["python", "javascript"])
            )
        
        logger.info(f"Initialized processors: {', '.join(processors.keys())}")
        return processors

    def _initialize_domain_solvers(self) -> Dict[str, Any]:
        solvers = {}
        domain_settings = self.settings.get_section("domains")
        enabled_domains = domain_settings.get("enabled", ["general"])
        
        solvers["general"] = GeneralSolver(self.llm, self.settings)
        
        if "math" in enabled_domains:
            from domains.math_solver import MathSolver
            solvers["math"] = MathSolver(self.llm, self.settings)
        
        if "code" in enabled_domains:
            from domains.code_analyzer import CodeAnalyzer
            solvers["code"] = CodeAnalyzer(self.llm, self.settings)
        
        if "science" in enabled_domains:
            from domains.science_solver import ScienceSolver
            solvers["science"] = ScienceSolver(self.llm, self.settings)
        
        logger.info(f"Initialized domain solvers: {', '.join(solvers.keys())}")
        return solvers

    def process_input(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        if input_type not in self.processors:
            logger.warning(f"Unsupported input type: {input_type}. Falling back to text.")
            input_type = "text"
            
        processor = self.processors[input_type]
        processed_input = processor.process(input_data)
        
        processed_input["timestamp"] = get_current_datetime()
        processed_input["input_type"] = input_type
        
        logger.debug(f"Processed {input_type} input: {processed_input}")
        return processed_input

    def detect_domain(self, processed_input: Dict[str, Any]) -> str:
        if not self.settings.get("domains", "auto_detect", True):
            return "general"
        
        content = processed_input.get("content", "")
        prompt = f"Determine the most appropriate domain for solving this problem. Options are: {', '.join(self.domain_solvers.keys())}. Problem: {content}"
        
        messages = [
            SystemMessage(content="You are an AI assistant that categorizes problems into domains."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.generate([messages])
        detected_domain = response.generations[0][0].text.strip().lower()
        
        for domain in self.domain_solvers.keys():
            if domain in detected_domain:
                detected_domain = domain
                break
        
        if detected_domain not in self.domain_solvers:
            logger.warning(f"Detected domain '{detected_domain}' not available. Falling back to general.")
            detected_domain = "general"
        
        logger.info(f"Detected domain: {detected_domain}")
        return detected_domain

    def solve_problem(self, input_data: Any, input_type: str = "text") -> Dict[str, Any]:
        processed_input = self.process_input(input_data, input_type)
        
        self.memory.add(processed_input, "input")
        
        domain = self.detect_domain(processed_input)
        
        self.current_problem = {
            "id": str(uuid.uuid4()),
            "input": processed_input,
            "domain": domain,
            "timestamp": get_current_datetime()
        }
        
        self.current_plan = self.planning.create_plan(processed_input, domain)
        self.current_step = 0
        
        domain_solver = self.domain_solvers[domain]
        solution = domain_solver.solve(processed_input, self.current_plan)
        
        solution["reasoning_steps"] = self.reasoning.get_reasoning_chain(processed_input, solution)
        
        solution["problem_id"] = self.current_problem["id"]
        solution["domain"] = domain
        solution["timestamp"] = get_current_datetime()
        
        self.memory.add(solution, "solution")
        self.solution_history.append(solution)
        
        logger.info(f"Problem solved. Domain: {domain}, Solution ID: {solution.get('id')}")
        return solution

    def refine_solution(self, feedback: str) -> Dict[str, Any]:
        if not self.current_problem or not self.solution_history:
            logger.warning("No current problem to refine")
            return {"error": "No current problem to refine"}
        
        processed_feedback = self.process_input(feedback)
        self.memory.add(processed_feedback, "feedback")
        
        last_solution = self.solution_history[-1]
        domain = last_solution["domain"]
        
        self.current_plan = self.planning.refine_plan(self.current_plan, processed_feedback)
        
        domain_solver = self.domain_solvers[domain]
        refined_solution = domain_solver.solve(self.current_problem["input"], self.current_plan, feedback=processed_feedback)
        
        refined_solution["reasoning_steps"] = self.reasoning.get_reasoning_chain(
            self.current_problem["input"], 
            refined_solution,
            feedback=processed_feedback
        )
        
        refined_solution["problem_id"] = self.current_problem["id"]
        refined_solution["domain"] = domain
        refined_solution["timestamp"] = get_current_datetime()
        refined_solution["is_refinement"] = True
        refined_solution["previous_solution_id"] = last_solution.get("id")
        
        self.memory.add(refined_solution, "solution")
        self.solution_history.append(refined_solution)
        
        logger.info(f"Solution refined. Domain: {domain}, Refined Solution ID: {refined_solution.get('id')}")
        return refined_solution

    def explain_solution(self, solution_id: Optional[str] = None) -> Dict[str, Any]:
        if solution_id:
            solution = next((s for s in self.solution_history if s.get("id") == solution_id), None)
            if not solution:
                solution = self.memory.get(solution_id)
        else:
            solution = self.solution_history[-1] if self.solution_history else None
        
        if not solution:
            logger.warning(f"Solution not found: {solution_id}")
            return {"error": "Solution not found"}
        
        explanation = self.reasoning.explain_solution(solution)
        
        explanation["solution_id"] = solution.get("id")
        explanation["timestamp"] = get_current_datetime()
        
        logger.info(f"Generated explanation for solution: {solution.get('id')}")
        return explanation

    def verify_solution(self, solution_id: Optional[str] = None) -> Dict[str, Any]:
        if solution_id:
            solution = next((s for s in self.solution_history if s.get("id") == solution_id), None)
            if not solution:
                solution = self.memory.get(solution_id)
        else:
            solution = self.solution_history[-1] if self.solution_history else None
        
        if not solution:
            logger.warning(f"Solution not found: {solution_id}")
            return {"error": "Solution not found"}
        
        domain = solution.get("domain", "general")
        domain_solver = self.domain_solvers.get(domain, self.domain_solvers["general"])
        
        verification = domain_solver.verify(solution)
        
        verification["solution_id"] = solution.get("id")
        verification["timestamp"] = get_current_datetime()
        
        logger.info(f"Verified solution: {solution.get('id')}, Result: {verification.get('is_correct')}")
        return verification

    def get_alternative_solution(self) -> Dict[str, Any]:
        if not self.current_problem:
            logger.warning("No current problem for alternative solution")
            return {"error": "No current problem"}
        
        alternative_plan = self.planning.create_alternative_plan(
            self.current_problem["input"],
            self.current_problem["domain"],
            self.current_plan
        )
        
        domain = self.current_problem["domain"]
        domain_solver = self.domain_solvers[domain]
        alternative_solution = domain_solver.solve(self.current_problem["input"], alternative_plan)
        
        alternative_solution["reasoning_steps"] = self.reasoning.get_reasoning_chain(
            self.current_problem["input"], 
            alternative_solution,
            is_alternative=True
        )
        
        alternative_solution["problem_id"] = self.current_problem["id"]
        alternative_solution["domain"] = domain
        alternative_solution["timestamp"] = get_current_datetime()
        alternative_solution["is_alternative"] = True
        
        self.memory.add(alternative_solution, "solution")
        self.solution_history.append(alternative_solution)
        
        logger.info(f"Generated alternative solution. Domain: {domain}, Solution ID: {alternative_solution.get('id')}")
        return alternative_solution

    def get_solution_history(self) -> List[Dict[str, Any]]:
        return self.solution_history

    def get_memory_context(self, query: str) -> List[Dict[str, Any]]:
        return self.memory.search(query)

    def reset(self) -> None:
        self.current_problem = None
        self.current_plan = None
        self.current_step = 0
        self.solution_history = []
        logger.info("Agent state reset")