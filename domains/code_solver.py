import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import tempfile
import os
import json

# Remove direct import of ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class CodeSolver:
    """
    A domain-specific solver for code-related problems.
    
    This solver can:
    1. Debug code issues
    2. Optimize code
    3. Explain code
    4. Generate code from requirements
    5. Translate code between languages
    6. Analyze code complexity and quality
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the code solver.
        Args:
            llm: Language model for code generation and analysis (any LangChain-compatible chat model)
        """
        self.llm = llm
        self.supported_languages = {
            'python': {
                'extension': '.py',
                'run_cmd': ['python', '{file}'],
                'comment': '#'
            },
            'javascript': {
                'extension': '.js',
                'run_cmd': ['node', '{file}'],
                'comment': '//'
            },
            'typescript': {
                'extension': '.ts',
                'run_cmd': ['ts-node', '{file}'],
                'comment': '//'
            },
            'java': {
                'extension': '.java',
                'run_cmd': ['java', '{file}'],
                'comment': '//'
            },
            'c': {
                'extension': '.c',
                'run_cmd': ['gcc', '{file}', '-o', '{output}', '&&', '{output}'],
                'comment': '//'
            },
            'cpp': {
                'extension': '.cpp',
                'run_cmd': ['g++', '{file}', '-o', '{output}', '&&', '{output}'],
                'comment': '//'
            },
            'csharp': {
                'extension': '.cs',
                'run_cmd': ['dotnet', 'run', '{file}'],
                'comment': '//'
            },
            'go': {
                'extension': '.go',
                'run_cmd': ['go', 'run', '{file}'],
                'comment': '//'
            },
            'ruby': {
                'extension': '.rb',
                'run_cmd': ['ruby', '{file}'],
                'comment': '#'
            },
            'php': {
                'extension': '.php',
                'run_cmd': ['php', '{file}'],
                'comment': '//'
            },
            'rust': {
                'extension': '.rs',
                'run_cmd': ['rustc', '{file}', '-o', '{output}', '&&', '{output}'],
                'comment': '//'
            },
            'shell': {
                'extension': '.sh',
                'run_cmd': ['bash', '{file}'],
                'comment': '#'
            },
            'sql': {
                'extension': '.sql',
                'run_cmd': ['sqlite3', '-init', '{file}'],
                'comment': '--'
            },
            'html': {
                'extension': '.html',
                'run_cmd': ['open', '{file}'],
                'comment': '<!--'
            },
            'css': {
                'extension': '.css',
                'run_cmd': [],
                'comment': '/*'
            }
        }
    
    def solve(self, problem: str, code: str = None, language: str = None, 
              context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve a code-related problem.
        
        Args:
            problem: The problem description
            code: The code to analyze/debug/optimize (optional)
            language: The programming language (optional)
            context: Additional context for the problem (optional)
            
        Returns:
            Dict containing the solution, explanation, and other relevant information
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None,
            'language': language,
            'execution_result': None,
            'error': None
        }
        
        try:
            # Detect problem type
            problem_type = self._detect_problem_type(problem)
            result['problem_type'] = problem_type
            
            # Detect language if not provided
            if code and not language:
                language = self._detect_language(code)
                result['language'] = language
            
            # Solve based on problem type
            if problem_type == 'debug':
                solution = self._debug_code(problem, code, language, context)
            elif problem_type == 'optimize':
                solution = self._optimize_code(problem, code, language, context)
            elif problem_type == 'explain':
                solution = self._explain_code(problem, code, language, context)
            elif problem_type == 'generate':
                solution = self._generate_code(problem, language, context)
            elif problem_type == 'translate':
                target_language = self._extract_target_language(problem)
                solution = self._translate_code(problem, code, language, target_language, context)
            elif problem_type == 'analyze':
                solution = self._analyze_code(problem, code, language, context)
            else:
                # Default to general code solution
                solution = self._solve_with_llm(problem, code, language, context)
            
            # Update result with solution
            result.update(solution)
            
            # Execute code if requested and if it's executable
            if code and context and context.get('execute_code', False) and language in self.supported_languages:
                execution_result = self._execute_code(result['code'] or code, language)
                result['execution_result'] = execution_result
            
        except Exception as e:
            logger.error(f"Error in code solver: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _detect_problem_type(self, problem: str) -> str:
        """
        Detect the type of code problem.
        
        Args:
            problem: The problem description
            
        Returns:
            str: The problem type (debug, optimize, explain, generate, translate, analyze)
        """
        problem_lower = problem.lower()
        
        if any(keyword in problem_lower for keyword in ['debug', 'fix', 'error', 'issue', 'not working']):
            return 'debug'
        elif any(keyword in problem_lower for keyword in ['optimize', 'improve', 'performance', 'faster', 'efficient']):
            return 'optimize'
        elif any(keyword in problem_lower for keyword in ['explain', 'understand', 'what does', 'how does']):
            return 'explain'
        elif any(keyword in problem_lower for keyword in ['generate', 'create', 'write', 'implement']):
            return 'generate'
        elif any(keyword in problem_lower for keyword in ['translate', 'convert', 'port', 'from', 'to language']):
            return 'translate'
        elif any(keyword in problem_lower for keyword in ['analyze', 'review', 'quality', 'complexity']):
            return 'analyze'
        else:
            return 'general'
    
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            str: The detected language
        """
        # Simple language detection based on keywords and syntax
        code_lower = code.lower()
        
        # Check for language-specific patterns
        if re.search(r'\bdef\b.*:|\bclass\b.*:|\bimport\b|\bfrom\b.*\bimport\b', code):
            return 'python'
        elif re.search(r'\bfunction\b|\bconst\b|\blet\b|\bvar\b|\b=>\b', code):
            if 'typescript' in code_lower or re.search(r':\s*(string|number|boolean|any)\b', code):
                return 'typescript'
            else:
                return 'javascript'
        elif re.search(r'\bpublic\b.*\bclass\b|\bprivate\b|\bprotected\b|\bimport\s+java\.', code):
            return 'java'
        elif re.search(r'#include\s*<.*\.h>|\bint\s+main\s*\(', code) and not re.search(r'\bclass\b', code):
            return 'c'
        elif re.search(r'#include\s*<.*>|\bstd::|\btemplate\b|\bnamespace\b', code):
            return 'cpp'
        elif re.search(r'\busing\s+System;|\bnamespace\b|\bpublic\s+class\b', code):
            return 'csharp'
        elif re.search(r'\bfunc\b|\bpackage\b\s+\w+', code):
            return 'go'
        elif re.search(r'\bdef\b|\bend\b|\bmodule\b|\brequire\b', code) and not re.search(r':', code):
            return 'ruby'
        elif re.search(r'<\?php|\becho\b|\bfunction\b.*\$', code):
            return 'php'
        elif re.search(r'\bfn\b|\blet\s+mut\b|\buse\s+std::', code):
            return 'rust'
        elif re.search(r'\bSELECT\b|\bFROM\b|\bWHERE\b|\bJOIN\b', code_lower):
            return 'sql'
        elif re.search(r'<!DOCTYPE\s+html>|<html>|<head>|<body>', code_lower):
            return 'html'
        elif re.search(r'\bbody\s*{|\b\.\w+\s*{|\b#\w+\s*{', code):
            return 'css'
        elif re.search(r'\becho\b|\bexport\b|\bsource\b', code):
            return 'shell'
        
        # Default to Python if can't determine
        return 'python'
    
    def _extract_target_language(self, problem: str) -> str:
        """
        Extract the target language for translation from the problem description.
        
        Args:
            problem: The problem description
            
        Returns:
            str: The target language
        """
        problem_lower = problem.lower()
        
        # Look for patterns like "translate to Python" or "convert from Java to C++"
        to_match = re.search(r'to\s+(\w+)', problem_lower)
        if to_match:
            target = to_match.group(1)
            
            # Map common language names to our supported languages
            language_map = {
                'python': 'python',
                'py': 'python',
                'javascript': 'javascript',
                'js': 'javascript',
                'typescript': 'typescript',
                'ts': 'typescript',
                'java': 'java',
                'c++': 'cpp',
                'cpp': 'cpp',
                'c#': 'csharp',
                'csharp': 'csharp',
                'go': 'go',
                'golang': 'go',
                'ruby': 'ruby',
                'php': 'php',
                'rust': 'rust',
                'sql': 'sql',
                'html': 'html',
                'css': 'css',
                'shell': 'shell',
                'bash': 'shell'
            }
            
            return language_map.get(target, 'python')
        
        return 'python'  # Default to Python if no target language found
    
    def _debug_code(self, problem: str, code: str, language: str, 
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Debug code issues.
        
        Args:
            problem: The problem description
            code: The code to debug
            language: The programming language
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None
        }
        
        try:
            # First, try to execute the code to get the error
            error_message = None
            if language in self.supported_languages:
                execution_result = self._execute_code(code, language)
                if execution_result.get('error'):
                    error_message = execution_result['error']
            
            # Use LLM to debug the code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for debugging
            messages = [
                SystemMessage(content="You are an expert code debugger. "
                                    "Fix the following code that has issues. "
                                    "Explain the problems you found and how you fixed them."),
                HumanMessage(content=f"Problem: {problem}\n\nCode:\n```{language}\n{code}\n```")
            ]
            
            # Add error message if available
            if error_message:
                messages.append(HumanMessage(content=f"Error message:\n{error_message}"))
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Extract fixed code
            code_pattern = re.compile(r'```(?:\w+)?\n(.+?)\n```', re.DOTALL)
            code_match = code_pattern.search(solution_text)
            
            if code_match:
                fixed_code = code_match.group(1).strip()
                result['code'] = fixed_code
            
            # Extract explanation and steps
            explanation_lines = []
            steps = []
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            in_explanation = True
            for line in solution_text.split('\n'):
                if line.strip().startswith('```'):
                    in_explanation = not in_explanation
                    continue
                
                if in_explanation:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':']:
                            steps.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
            
            result['explanation'] = ' '.join(explanation_lines)
            result['steps'] = steps
            result['solution'] = "Fixed code with explanation"
            
        except Exception as e:
            logger.error(f"Error debugging code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _optimize_code(self, problem: str, code: str, language: str, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize code for better performance or readability.
        
        Args:
            problem: The problem description
            code: The code to optimize
            language: The programming language
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None
        }
        
        try:
            # Use LLM to optimize the code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Determine optimization focus
            problem_lower = problem.lower()
            if 'performance' in problem_lower or 'faster' in problem_lower or 'efficient' in problem_lower:
                optimization_focus = 'performance'
            elif 'readability' in problem_lower or 'clean' in problem_lower or 'maintainable' in problem_lower:
                optimization_focus = 'readability'
            else:
                optimization_focus = 'both performance and readability'
            
            # Create prompt for optimization
            messages = [
                SystemMessage(content=f"You are an expert code optimizer. "
                                    f"Optimize the following code for {optimization_focus}. "
                                    f"Explain the optimizations you made and why they improve the code."),
                HumanMessage(content=f"Problem: {problem}\n\nCode:\n```{language}\n{code}\n```")
            ]
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Extract optimized code
            code_pattern = re.compile(r'```(?:\w+)?\n(.+?)\n```', re.DOTALL)
            code_match = code_pattern.search(solution_text)
            
            if code_match:
                optimized_code = code_match.group(1).strip()
                result['code'] = optimized_code
            
            # Extract explanation and steps
            explanation_lines = []
            steps = []
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            in_explanation = True
            for line in solution_text.split('\n'):
                if line.strip().startswith('```'):
                    in_explanation = not in_explanation
                    continue
                
                if in_explanation:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':']:
                            steps.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
            
            result['explanation'] = ' '.join(explanation_lines)
            result['steps'] = steps
            result['solution'] = f"Optimized code for {optimization_focus}"
            
        except Exception as e:
            logger.error(f"Error optimizing code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _explain_code(self, problem: str, code: str, language: str, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Explain code functionality.
        
        Args:
            problem: The problem description
            code: The code to explain
            language: The programming language
            context: Additional context
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': code
        }
        
        try:
            # Use LLM to explain the code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content="You are an expert code explainer. "
                                    "Explain the following code in detail. "
                                    "Break down the explanation into logical sections and describe what each part does."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Add problem context if available
            if problem:
                messages.append(HumanMessage(content=f"Context: {problem}"))
            
            # Add additional context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract sections and steps
            sections = []
            explanation_lines = []
            
            current_section = None
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
                    else:
                        explanation_lines.append(line.strip())
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = '\n\n'.join(explanation_lines)
            result['steps'] = sections
            result['solution'] = "Code explanation"
            
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _generate_code(self, problem: str, language: str = None, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate code from requirements.
        
        Args:
            problem: The problem description/requirements
            language: The target programming language
            context: Additional context
            
        Returns:
            Dict containing the generated code and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None
        }
        
        try:
            # Default to Python if no language specified
            if not language:
                language = 'python'
            
            # Use LLM to generate code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for code generation
            messages = [
                SystemMessage(content=f"You are an expert {language} programmer. "
                                    f"Generate code based on the following requirements. "
                                    f"Provide a detailed explanation of how the code works."),
                HumanMessage(content=f"Requirements: {problem}")
            ]
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Extract generated code
            code_pattern = re.compile(r'```(?:\w+)?\n(.+?)\n```', re.DOTALL)
            code_match = code_pattern.search(solution_text)
            
            if code_match:
                generated_code = code_match.group(1).strip()
                result['code'] = generated_code
            
            # Extract explanation and steps
            explanation_lines = []
            steps = []
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            in_explanation = True
            for line in solution_text.split('\n'):
                if line.strip().startswith('```'):
                    in_explanation = not in_explanation
                    continue
                
                if in_explanation:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':']:
                            steps.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
            
            result['explanation'] = ' '.join(explanation_lines)
            result['steps'] = steps
            result['solution'] = f"Generated {language} code"
            
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _translate_code(self, problem: str, code: str, source_language: str, 
                       target_language: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Translate code from one language to another.
        
        Args:
            problem: The problem description
            code: The code to translate
            source_language: The source programming language
            target_language: The target programming language
            context: Additional context
            
        Returns:
            Dict containing the translated code and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None,
            'language': target_language
        }
        
        try:
            # Use LLM to translate the code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for translation
            messages = [
                SystemMessage(content=f"You are an expert programmer fluent in both {source_language} and {target_language}. "
                                    f"Translate the following {source_language} code to {target_language}. "
                                    f"Explain the key differences and any language-specific adaptations you made."),
                HumanMessage(content=f"Code to translate:\n```{source_language}\n{code}\n```")
            ]
            
            # Add problem context if available
            if problem:
                messages.append(HumanMessage(content=f"Context: {problem}"))
            
            # Add additional context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate translation
            response = self.llm.generate([messages])
            translation_text = response.generations[0][0].text.strip()
            
            # Extract translated code
            code_pattern = re.compile(r'```(?:\w+)?\n(.+?)\n```', re.DOTALL)
            code_match = code_pattern.search(translation_text)
            
            if code_match:
                translated_code = code_match.group(1).strip()
                result['code'] = translated_code
            
            # Extract explanation and steps
            explanation_lines = []
            steps = []
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            in_explanation = True
            for line in translation_text.split('\n'):
                if line.strip().startswith('```'):
                    in_explanation = not in_explanation
                    continue
                
                if in_explanation:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':']:
                            steps.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
            
            result['explanation'] = ' '.join(explanation_lines)
            result['steps'] = steps
            result['solution'] = f"Translated from {source_language} to {target_language}"
            
        except Exception as e:
            logger.error(f"Error translating code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _analyze_code(self, problem: str, code: str, language: str, 
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze code quality, complexity, and potential issues.
        
        Args:
            problem: The problem description
            code: The code to analyze
            language: The programming language
            context: Additional context
            
        Returns:
            Dict containing the analysis results
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': code,
            'analysis': {
                'complexity': None,
                'quality': None,
                'issues': [],
                'suggestions': []
            }
        }
        
        try:
            # Use LLM to analyze the code
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for analysis
            messages = [
                SystemMessage(content=f"You are an expert code reviewer and analyzer. "
                                    f"Analyze the following {language} code for quality, complexity, and potential issues. "
                                    f"Provide a detailed analysis with specific suggestions for improvement."),
                HumanMessage(content=f"Code to analyze:\n```{language}\n{code}\n```")
            ]
            
            # Add problem context if available
            if problem:
                messages.append(HumanMessage(content=f"Context: {problem}"))
            
            # Add additional context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate analysis
            response = self.llm.generate([messages])
            analysis_text = response.generations[0][0].text.strip()
            
            # Extract complexity, quality, issues, and suggestions
            complexity_match = re.search(r'(?i)complexity[:\s]+(\w+)', analysis_text)
            quality_match = re.search(r'(?i)quality[:\s]+(\w+)', analysis_text)
            
            if complexity_match:
                result['analysis']['complexity'] = complexity_match.group(1).strip()
            
            if quality_match:
                result['analysis']['quality'] = quality_match.group(1).strip()
            
            # Extract issues and suggestions
            issues = []
            suggestions = []
            
            in_issues = False
            in_suggestions = False
            
            for line in analysis_text.split('\n'):
                if re.search(r'(?i)issues?[:\s]+', line):
                    in_issues = True
                    in_suggestions = False
                    continue
                elif re.search(r'(?i)suggestions?[:\s]+', line):
                    in_issues = False
                    in_suggestions = True
                    continue
                elif line.strip() and (line.strip()[0] == '#' or line.strip().endswith(':')):
                    in_issues = False
                    in_suggestions = False
                
                if in_issues and line.strip() and line.strip()[0] in ['-', '*', '•']:
                    issues.append(line.strip()[1:].strip())
                elif in_suggestions and line.strip() and line.strip()[0] in ['-', '*', '•']:
                    suggestions.append(line.strip()[1:].strip())
            
            result['analysis']['issues'] = issues
            result['analysis']['suggestions'] = suggestions
            
            # Generate overall explanation
            explanation = f"Code analysis for {language} code. "
            if result['analysis']['complexity']:
                explanation += f"Complexity: {result['analysis']['complexity']}. "
            if result['analysis']['quality']:
                explanation += f"Quality: {result['analysis']['quality']}. "
            explanation += f"Found {len(issues)} issues and provided {len(suggestions)} suggestions for improvement."
            
            result['explanation'] = explanation
            result['solution'] = "Code analysis complete"
            
        except Exception as e:
            logger.error(f"Error analyzing code: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _solve_with_llm(self, problem: str, code: str = None, language: str = None, 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a general code problem using the language model.
        
        Args:
            problem: The problem description
            code: The code (optional)
            language: The programming language (optional)
            context: Additional context (optional)
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'code': None
        }
        
        try:
            # Use LLM to solve the problem
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for problem-solving
            system_content = "You are an expert programmer and problem solver. "
            if language:
                system_content += f"You specialize in {language} programming. "
            system_content += "Solve the following code-related problem with detailed explanations."
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"Problem: {problem}")
            ]
            
            # Add code if available
            if code:
                code_str = f"\n\nCode:\n```{language or ''}\n{code}\n```"
                messages[1] = HumanMessage(content=messages[1].content + code_str)
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    if key != 'execute_code':  # Skip execution flag
                        context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Extract code if present
            code_pattern = re.compile(r'```(?:\w+)?\n(.+?)\n```', re.DOTALL)
            code_match = code_pattern.search(solution_text)
            
            if code_match:
                solution_code = code_match.group(1).strip()
                result['code'] = solution_code
            
            # Extract explanation and steps
            explanation_lines = []
            steps = []
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            in_explanation = True
            for line in solution_text.split('\n'):
                if line.strip().startswith('```'):
                    in_explanation = not in_explanation
                    continue
                
                if in_explanation:
                    if line.strip():
                        if line.strip().startswith('-') or line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':']:
                            steps.append(line.strip())
                        else:
                            explanation_lines.append(line.strip())
            
            result['explanation'] = ' '.join(explanation_lines)
            result['steps'] = steps
            result['solution'] = "Solution generated"
            
        except Exception as e:
            logger.error(f"Error solving with LLM: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _execute_code(self, code: str, language: str) -> Dict[str, Any]:
        """
        Execute code and return the result.
        
        Args:
            code: The code to execute
            language: The programming language
            
        Returns:
            Dict containing the execution result and any errors
        """
        result = {
            'output': None,
            'error': None,
            'execution_time': None
        }
        
        if language not in self.supported_languages:
            result['error'] = f"Language {language} is not supported for execution"
            return result
        
        try:
            import time
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix=self.supported_languages[language]['extension'], 
                                            delete=False, mode='w') as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            # Prepare the command
            output_path = temp_file_path + '.out'
            cmd = []
            for part in self.supported_languages[language]['run_cmd']:
                if part == '{file}':
                    cmd.append(temp_file_path)
                elif part == '{output}':
                    cmd.append(output_path)
                else:
                    cmd.append(part)
            
            # Execute the code
            start_time = time.time()
            
            # Handle commands with '&&'
            if '&&' in cmd:
                idx = cmd.index('&&')
                first_cmd = cmd[:idx]
                second_cmd = cmd[idx+1:]
                
                # Run first command
                process = subprocess.run(first_cmd, capture_output=True, text=True, timeout=10)
                if process.returncode != 0:
                    result['error'] = process.stderr
                    return result
                
                # Run second command
                process = subprocess.run(second_cmd, capture_output=True, text=True, timeout=10)
            else:
                process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            end_time = time.time()
            
            # Process the result
            result['execution_time'] = end_time - start_time
            
            if process.returncode == 0:
                result['output'] = process.stdout
            else:
                result['error'] = process.stderr
            
        except subprocess.TimeoutExpired:
            result['error'] = "Execution timed out after 10 seconds"
        except Exception as e:
            result['error'] = str(e)
        finally:
            # Clean up temporary files
            try:
                os.remove(temp_file_path)
                if os.path.exists(output_path):
                    os.remove(output_path)
            except:
                pass
        
        return result
    
    def verify_solution(self, problem: str, code: str, language: str, 
                       expected_output: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify if a code solution is correct.
        
        Args:
            problem: The problem description
            code: The code solution
            language: The programming language
            expected_output: The expected output (optional)
            
        Returns:
            Dict containing verification results
        """
        result = {
            'is_correct': False,
            'passes_execution': False,
            'matches_expected_output': False,
            'error': None
        }
        
        try:
            # Check if code executes without errors
            if language in self.supported_languages:
                execution_result = self._execute_code(code, language)
                result['passes_execution'] = execution_result.get('error') is None
                
                # Check if output matches expected output
                if expected_output and execution_result.get('output'):
                    # Normalize outputs for comparison (remove whitespace, case insensitive)
                    normalized_expected = re.sub(r'\s+', '', expected_output.lower())
                    normalized_actual = re.sub(r'\s+', '', execution_result['output'].lower())
                    result['matches_expected_output'] = normalized_expected in normalized_actual
            
            # Use LLM to verify solution correctness
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for verification
            messages = [
                SystemMessage(content="You are an expert code reviewer. "
                                    "Verify if the following code correctly solves the given problem. "
                                    "Provide a yes/no answer and explain your reasoning."),
                HumanMessage(content=f"Problem: {problem}\n\nCode Solution:\n```{language}\n{code}\n```")
            ]
            
            # Add execution results if available
            if language in self.supported_languages:
                execution_str = "\n\nExecution Results:\n"
                if execution_result.get('error'):
                    execution_str += f"Error: {execution_result['error']}\n"
                else:
                    execution_str += f"Output: {execution_result.get('output', 'No output')}\n"
                    execution_str += f"Execution Time: {execution_result.get('execution_time', 0):.4f} seconds\n"
                
                messages.append(HumanMessage(content=execution_str))
            
            # Add expected output if available
            if expected_output:
                messages.append(HumanMessage(content=f"Expected Output: {expected_output}"))
            
            # Generate verification
            response = self.llm.generate([messages])
            verification_text = response.generations[0][0].text.strip().lower()
            
            # Check if the LLM thinks the solution is correct
            result['is_correct'] = ('yes' in verification_text[:50] or 
                                   'correct' in verification_text[:50] or 
                                   'solves the problem' in verification_text[:50])
            
            # If execution passes and either matches expected output or LLM says it's correct
            if result['passes_execution'] and (result['matches_expected_output'] or result['is_correct']):
                result['is_correct'] = True
            
        except Exception as e:
            logger.error(f"Error verifying solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, code: str, language: str, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a code solution.
        
        Args:
            code: The code to explain
            language: The programming language
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'sections': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of what the code does.",
                'medium': "Explain the main components and logic of the code.",
                'high': "Provide a detailed line-by-line explanation of the code."
            }.get(detail_level.lower(), "Explain the main components and logic of the code.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert code explainer. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational."),
                HumanMessage(content=f"Code to explain:\n```{language}\n{code}\n```")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract sections
            sections = []
            current_section = None
            
            for line in explanation_text.split('\n'):
                if line.strip():
                    if re.match(r'^#+\s+', line) or line.strip().endswith(':'):
                        # This is a section header
                        if current_section:
                            sections.append(current_section)
                        current_section = line.strip()
                    elif current_section:
                        current_section += '\n' + line.strip()
            
            if current_section:
                sections.append(current_section)
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['sections'] = sections
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    

            
            # Generate explanation