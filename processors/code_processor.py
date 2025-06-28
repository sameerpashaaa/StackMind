#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Code Processor Module

This module implements code processing capabilities for the AI Problem Solver,
allowing it to handle and analyze code-based inputs across various programming languages.
"""

import logging
import re
import os
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class CodeProcessor:
    """
    Code processor for handling code-based inputs.
    
    This processor analyzes and extracts information from code inputs,
    including syntax highlighting, structure analysis, and execution.
    
    Attributes:
        llm: Optional language model for enhanced code understanding
        settings: Application settings
    """
    
    def __init__(self, llm=None, settings=None):
        """
        Initialize the Code Processor.
        
        Args:
            llm: Optional language model for enhanced code understanding
            settings: Optional application settings
        """
        self.llm = llm
        self.settings = settings
        
        # Language detection patterns
        self.language_patterns = self._initialize_language_patterns()
        
        logger.info("Code processor initialized")
    
    def process(self, code: str) -> Dict[str, Any]:
        """
        Process code input and extract relevant information.
        
        Args:
            code: The code input to process
            
        Returns:
            Dict[str, Any]: Processed code with metadata and analysis
        """
        # Initialize result
        result = {
            "content": code,
            "content_type": "code",
            "metadata": {}
        }
        
        # Extract basic metadata
        result["metadata"]["length"] = len(code)
        result["metadata"]["line_count"] = code.count('\n') + 1
        
        # Detect language
        detected_language = self._detect_language(code)
        result["metadata"]["language"] = detected_language
        
        # Analyze code structure
        result["structure"] = self._analyze_structure(code, detected_language)
        
        # Extract imports/dependencies
        result["dependencies"] = self._extract_dependencies(code, detected_language)
        
        # Check for syntax errors
        result["syntax_check"] = self._check_syntax(code, detected_language)
        
        # Generate documentation if LLM is available
        if self.llm:
            result["documentation"] = self._generate_documentation(code, detected_language)
        
        logger.debug(f"Processed code input: {len(code)} chars, {result['metadata']['line_count']} lines, language: {detected_language}")
        return result
    
    def _initialize_language_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize patterns for language detection and analysis.
        
        Returns:
            Dict[str, Dict[str, Any]]: Language patterns
        """
        return {
            "python": {
                "extensions": [".py", ".pyw", ".pyc", ".pyo", ".pyd"],
                "patterns": [
                    r'^#!/usr/bin/env\s+python',
                    r'^# -\*- coding: .+ -\*-',
                    r'^import\s+[\w\.]+',
                    r'^from\s+[\w\.]+\s+import',
                    r'def\s+\w+\s*\(',
                    r'class\s+\w+[\(\:]'
                ],
                "comment": ["#"],
                "function": r'def\s+(\w+)\s*\(',
                "class": r'class\s+(\w+)[\(\:]',
                "import": [r'^import\s+([\w\.]+)', r'^from\s+([\w\.]+)\s+import']
            },
            "javascript": {
                "extensions": [".js", ".jsx", ".mjs"],
                "patterns": [
                    r'function\s+\w+\s*\(',
                    r'const\s+\w+\s*=',
                    r'let\s+\w+\s*=',
                    r'var\s+\w+\s*=',
                    r'=>',
                    r'document\.\w+',
                    r'window\.\w+',
                    r'\$\(.*\)'
                ],
                "comment": ["//", "/*"],
                "function": [r'function\s+(\w+)\s*\(', r'const\s+(\w+)\s*=\s*function', r'const\s+(\w+)\s*=\s*\(.*\)\s*=>'],
                "class": r'class\s+(\w+)',
                "import": [r'import\s+.*\s+from\s+[\'"]([\w\.\-\/]+)[\'"]', r'require\([\'"]([\w\.\-\/]+)[\'"]\)']
            },
            "java": {
                "extensions": [".java", ".class"],
                "patterns": [
                    r'public\s+class',
                    r'private\s+class',
                    r'protected\s+class',
                    r'public\s+static\s+void\s+main',
                    r'import\s+java\.',
                    r'package\s+[\w\.]+;'
                ],
                "comment": ["//", "/*"],
                "function": r'(public|private|protected|static|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\(',
                "class": r'(public|private|protected)\s+class\s+(\w+)',
                "import": r'import\s+([\w\.\*]+);'
            },
            "c": {
                "extensions": [".c", ".h"],
                "patterns": [
                    r'#include\s+[<"]\w+\.h[>"]',
                    r'int\s+main\s*\(',
                    r'void\s+\w+\s*\(',
                    r'struct\s+\w+\s*\{',
                    r'typedef'
                ],
                "comment": ["//", "/*"],
                "function": r'[\w\*]+\s+(\w+)\s*\(',
                "struct": r'struct\s+(\w+)',
                "include": r'#include\s+[<"]([\w\.]+)[>"]'
            },
            "cpp": {
                "extensions": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
                "patterns": [
                    r'#include\s+[<"]\w+(\.h|\.hpp)[>"]',
                    r'namespace\s+\w+',
                    r'class\s+\w+\s*(:\s*\w+)?\s*\{',
                    r'template\s*<',
                    r'std::'
                ],
                "comment": ["//", "/*"],
                "function": r'[\w\*\:~]+\s+(\w+)\s*\(',
                "class": r'class\s+(\w+)',
                "include": r'#include\s+[<"]([\w\.]+)[>"]',
                "namespace": r'namespace\s+(\w+)'
            },
            "csharp": {
                "extensions": [".cs"],
                "patterns": [
                    r'using\s+[\w\.]+;',
                    r'namespace\s+[\w\.]+',
                    r'class\s+\w+',
                    r'public\s+static\s+void\s+Main'
                ],
                "comment": ["//", "/*"],
                "function": r'(public|private|protected|internal|static|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\(',
                "class": r'(public|private|protected|internal|static|\s)+class\s+(\w+)',
                "using": r'using\s+([\w\.]+);'
            },
            "ruby": {
                "extensions": [".rb", ".rbw"],
                "patterns": [
                    r'^#!/usr/bin/env\s+ruby',
                    r'require\s+[\'"][\w\.\-\/]+[\'"]',
                    r'def\s+\w+',
                    r'class\s+\w+',
                    r'module\s+\w+'
                ],
                "comment": ["#"],
                "function": r'def\s+(\w+)',
                "class": r'class\s+(\w+)',
                "require": r'require\s+[\'"]([\w\.\-\/]+)[\'"]'
            },
            "go": {
                "extensions": [".go"],
                "patterns": [
                    r'package\s+\w+',
                    r'import\s+\(',
                    r'import\s+"',
                    r'func\s+\w+',
                    r'type\s+\w+\s+struct'
                ],
                "comment": ["//", "/*"],
                "function": r'func\s+(\w+)',
                "struct": r'type\s+(\w+)\s+struct',
                "import": r'import\s+[\(\s]*"([\w\.\-\/]+)"'
            },
            "php": {
                "extensions": [".php", ".phtml", ".php3", ".php4", ".php5", ".php7", ".phps"],
                "patterns": [
                    r'<\?php',
                    r'function\s+\w+\s*\(',
                    r'class\s+\w+',
                    r'namespace\s+[\w\\]+',
                    r'use\s+[\w\\]+'
                ],
                "comment": ["//", "#", "/*"],
                "function": r'function\s+(\w+)\s*\(',
                "class": r'class\s+(\w+)',
                "namespace": r'namespace\s+([\w\\]+)',
                "use": r'use\s+([\w\\]+)'
            },
            "html": {
                "extensions": [".html", ".htm", ".xhtml"],
                "patterns": [
                    r'<!DOCTYPE\s+html',
                    r'<html',
                    r'<head',
                    r'<body',
                    r'<script',
                    r'<style',
                    r'<div',
                    r'<span',
                    r'<a\s+href'
                ],
                "comment": ["<!--"],
                "tag": r'<(\w+)'
            },
            "css": {
                "extensions": [".css"],
                "patterns": [
                    r'\w+\s*\{',
                    r'#\w+\s*\{',
                    r'\.\w+\s*\{',
                    r'@media',
                    r'@import',
                    r'@keyframes'
                ],
                "comment": ["/*"],
                "selector": r'([\w\.#][-\w\.#:]+)\s*\{',
                "property": r'\s*([-\w]+)\s*:'
            },
            "sql": {
                "extensions": [".sql"],
                "patterns": [
                    r'SELECT\s+.+\s+FROM',
                    r'INSERT\s+INTO',
                    r'UPDATE\s+.+\s+SET',
                    r'DELETE\s+FROM',
                    r'CREATE\s+TABLE',
                    r'ALTER\s+TABLE',
                    r'DROP\s+TABLE',
                    r'EXEC\s+\w+',
                    r'EXECUTE\s+\w+'
                ],
                "comment": ["--", "/*"],
                "table": [r'FROM\s+([\w\.]+)', r'JOIN\s+([\w\.]+)', r'INTO\s+([\w\.]+)', r'UPDATE\s+([\w\.]+)', r'TABLE\s+([\w\.]+)'],
                "function": r'CREATE\s+FUNCTION\s+([\w\.]+)'
            },
            "shell": {
                "extensions": [".sh", ".bash", ".zsh", ".ksh"],
                "patterns": [
                    r'^#!/bin/(ba)?sh',
                    r'^#!/usr/bin/(ba)?sh',
                    r'export\s+\w+=',
                    r'function\s+\w+\s*\{',
                    r'\w+\s*\(\)\s*\{',
                    r'if\s+\[\s+.+\s+\]',
                    r'for\s+\w+\s+in'
                ],
                "comment": ["#"],
                "function": [r'function\s+(\w+)', r'(\w+)\s*\(\)\s*\{'],
                "variable": r'\$(\w+)'
            },
            "powershell": {
                "extensions": [".ps1", ".psm1", ".psd1"],
                "patterns": [
                    r'function\s+\w+\-\w+',
                    r'\$\w+\s*=',
                    r'Write\-Host',
                    r'Get\-\w+',
                    r'Set\-\w+',
                    r'New\-\w+',
                    r'Remove\-\w+'
                ],
                "comment": ["#"],
                "function": r'function\s+(\w+\-\w+)',
                "cmdlet": r'(\w+\-\w+)'
            },
            "rust": {
                "extensions": [".rs"],
                "patterns": [
                    r'fn\s+\w+\s*\(',
                    r'struct\s+\w+',
                    r'enum\s+\w+',
                    r'impl\s+\w+',
                    r'use\s+[\w\:]+',
                    r'mod\s+\w+',
                    r'pub\s+'
                ],
                "comment": ["//", "/*"],
                "function": r'fn\s+(\w+)',
                "struct": r'struct\s+(\w+)',
                "enum": r'enum\s+(\w+)',
                "use": r'use\s+([\w\:]+)'
            },
            "typescript": {
                "extensions": [".ts", ".tsx"],
                "patterns": [
                    r'function\s+\w+\s*\(',
                    r'const\s+\w+\s*:',
                    r'let\s+\w+\s*:',
                    r'var\s+\w+\s*:',
                    r'interface\s+\w+',
                    r'class\s+\w+',
                    r'import\s+.*\s+from',
                    r'export'
                ],
                "comment": ["//", "/*"],
                "function": [r'function\s+(\w+)\s*\(', r'const\s+(\w+)\s*=\s*function', r'const\s+(\w+)\s*=\s*\(.*\)\s*=>'],
                "class": r'class\s+(\w+)',
                "interface": r'interface\s+(\w+)',
                "import": r'import\s+.*\s+from\s+[\'"]([\w\.\-\/]+)[\'"]'
            }
        }
    
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of the code.
        
        Args:
            code: The code to analyze
            
        Returns:
            str: Detected programming language
        """
        # Check for file extension in the first line (e.g., from a code fence)
        first_line = code.split('\n', 1)[0].strip()
        fence_match = re.match(r'^```(\w+)', first_line)
        if fence_match:
            lang = fence_match.group(1).lower()
            # Map common aliases
            lang_map = {
                'js': 'javascript',
                'py': 'python',
                'rb': 'ruby',
                'cs': 'csharp',
                'ts': 'typescript',
                'bash': 'shell',
                'sh': 'shell'
            }
            return lang_map.get(lang, lang)
        
        # Score each language based on pattern matches
        scores = {}
        for lang, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns.get("patterns", []):
                if re.search(pattern, code, re.MULTILINE):
                    score += 1
            scores[lang] = score
        
        # Get the language with the highest score
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                # If there's a tie, prefer more specific languages
                max_langs = [lang for lang, score in scores.items() if score == max_score]
                if len(max_langs) > 1:
                    # Preference order for common languages
                    preference = ['python', 'javascript', 'typescript', 'java', 'cpp', 'csharp', 'go', 'rust', 'php', 'ruby']
                    for lang in preference:
                        if lang in max_langs:
                            return lang
                return max_langs[0]
        
        # Default to plaintext if no language is detected
        return "plaintext"
    
    def _analyze_structure(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze the structure of the code.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            Dict[str, Any]: Code structure analysis
        """
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "variables": []
        }
        
        # Get language patterns
        lang_patterns = self.language_patterns.get(language, {})
        if not lang_patterns:
            return structure
        
        # Extract functions
        function_patterns = lang_patterns.get("function", [])
        if not isinstance(function_patterns, list):
            function_patterns = [function_patterns]
        
        for pattern in function_patterns:
            if pattern:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    function_name = match.group(1)
                    if function_name not in structure["functions"]:
                        structure["functions"].append(function_name)
        
        # Extract classes
        class_pattern = lang_patterns.get("class")
        if class_pattern:
            for match in re.finditer(class_pattern, code, re.MULTILINE):
                class_name = match.group(1) if len(match.groups()) >= 1 else match.group(2)
                if class_name not in structure["classes"]:
                    structure["classes"].append(class_name)
        
        # Extract imports/dependencies
        import_patterns = lang_patterns.get("import", [])
        if not isinstance(import_patterns, list):
            import_patterns = [import_patterns]
        
        for pattern in import_patterns:
            if pattern:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    import_name = match.group(1)
                    if import_name not in structure["imports"]:
                        structure["imports"].append(import_name)
        
        # Extract variables (basic implementation, can be enhanced)
        if language == "python":
            # Match variable assignments
            for match in re.finditer(r'^(\w+)\s*=\s*(?!\s*lambda)', code, re.MULTILINE):
                var_name = match.group(1)
                if var_name not in structure["variables"] and var_name not in structure["functions"]:
                    structure["variables"].append(var_name)
        elif language in ["javascript", "typescript"]:
            # Match variable declarations
            for match in re.finditer(r'(const|let|var)\s+(\w+)\s*=', code, re.MULTILINE):
                var_name = match.group(2)
                if var_name not in structure["variables"]:
                    structure["variables"].append(var_name)
        
        return structure
    
    def _extract_dependencies(self, code: str, language: str) -> List[str]:
        """
        Extract dependencies from the code.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            List[str]: Extracted dependencies
        """
        dependencies = []
        
        # Get language patterns
        lang_patterns = self.language_patterns.get(language, {})
        if not lang_patterns:
            return dependencies
        
        # Extract imports/dependencies
        import_patterns = lang_patterns.get("import", [])
        if not isinstance(import_patterns, list):
            import_patterns = [import_patterns]
        
        for pattern in import_patterns:
            if pattern:
                for match in re.finditer(pattern, code, re.MULTILINE):
                    import_name = match.group(1)
                    # Clean up the import name
                    import_name = import_name.split('.')[0]  # Get the top-level package
                    if import_name and import_name not in dependencies:
                        dependencies.append(import_name)
        
        # Special handling for specific languages
        if language == "python":
            # Check for pip installable packages
            common_packages = {
                'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 'tensorflow', 'torch', 'keras',
                'django', 'flask', 'requests', 'beautifulsoup4', 'selenium', 'pillow', 'opencv',
                'nltk', 'spacy', 'gensim', 'transformers', 'fastapi', 'sqlalchemy', 'pymongo',
                'pytest', 'unittest', 'logging', 'argparse', 'os', 'sys', 're', 'json', 'csv',
                'time', 'datetime', 'random', 'math', 'collections', 'itertools', 'functools'
            }
            dependencies = [dep for dep in dependencies if dep.lower() in common_packages]
        
        return dependencies
    
    def _check_syntax(self, code: str, language: str) -> Dict[str, Any]:
        """
        Check the code for syntax errors.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            Dict[str, Any]: Syntax check results
        """
        result = {
            "valid": True,
            "errors": []
        }
        
        # Python syntax checking
        if language == "python":
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                result["valid"] = False
                result["errors"].append({
                    "line": e.lineno,
                    "column": e.offset,
                    "message": str(e)
                })
        # JavaScript/TypeScript basic syntax checking
        elif language in ["javascript", "typescript"]:
            # Check for mismatched brackets/parentheses
            brackets = {'(': ')', '[': ']', '{': '}'}
            stack = []
            for i, char in enumerate(code):
                if char in brackets.keys():
                    stack.append((char, i))
                elif char in brackets.values():
                    if not stack or brackets[stack[-1][0]] != char:
                        line = code[:i].count('\n') + 1
                        col = i - code[:i].rfind('\n')
                        result["valid"] = False
                        result["errors"].append({
                            "line": line,
                            "column": col,
                            "message": f"Mismatched bracket: '{char}'"
                        })
                        break
                    stack.pop()
            
            if stack:
                char, pos = stack[-1]
                line = code[:pos].count('\n') + 1
                col = pos - code[:pos].rfind('\n')
                result["valid"] = False
                result["errors"].append({
                    "line": line,
                    "column": col,
                    "message": f"Unclosed bracket: '{char}'"
                })
        
        return result
    
    def _generate_documentation(self, code: str, language: str) -> str:
        """
        Generate documentation for the code using a language model.
        
        Args:
            code: The code to document
            language: The programming language
            
        Returns:
            str: Generated documentation
        """
        if not self.llm:
            return ""
        
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for documentation
            messages = [
                SystemMessage(content=f"You are an expert {language} developer. "
                                    "Generate comprehensive documentation for the following code. "
                                    "Include an overview, function/class descriptions, parameters, return values, "
                                    "and usage examples where appropriate."),
                HumanMessage(content=f"```{language}\n{code}\n```")
            ]
            
            # Generate documentation
            response = self.llm.generate([messages])
            documentation = response.generations[0][0].text.strip()
            
            return documentation
            
        except Exception as e:
            logger.warning(f"Error generating code documentation: {str(e)}")
            return ""
    
    def execute_code(self, code: str, language: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Execute code and return the result.
        
        This method should be used with caution and only with trusted code.
        
        Args:
            code: The code to execute
            language: The programming language
            timeout: Maximum execution time in seconds
            
        Returns:
            Dict[str, Any]: Execution results
        """
        result = {
            "success": False,
            "output": "",
            "error": ""
        }
        
        # Check if code execution is allowed in settings
        if self.settings and hasattr(self.settings, 'code_execution') and \
           not self.settings.code_execution.get('enabled', False):
            result["error"] = "Code execution is disabled in settings"
            return result
        
        # Python code execution
        if language == "python":
            try:
                import subprocess
                import tempfile
                import signal
                import os
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(code.encode('utf-8'))
                
                # Execute the code in a separate process with timeout
                process = subprocess.Popen(
                    ["python", temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    result["success"] = process.returncode == 0
                    result["output"] = stdout
                    result["error"] = stderr
                except subprocess.TimeoutExpired:
                    # Kill the process if it times out
                    process.kill()
                    result["error"] = f"Execution timed out after {timeout} seconds"
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
            except Exception as e:
                result["error"] = f"Error executing Python code: {str(e)}"
        
        # JavaScript code execution (requires Node.js)
        elif language == "javascript":
            try:
                import subprocess
                import tempfile
                import os
                
                # Check if Node.js is installed
                try:
                    subprocess.run(["node", "--version"], check=True, capture_output=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    result["error"] = "Node.js is not installed or not in PATH"
                    return result
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix='.js', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(code.encode('utf-8'))
                
                # Execute the code with Node.js
                process = subprocess.Popen(
                    ["node", temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    result["success"] = process.returncode == 0
                    result["output"] = stdout
                    result["error"] = stderr
                except subprocess.TimeoutExpired:
                    # Kill the process if it times out
                    process.kill()
                    result["error"] = f"Execution timed out after {timeout} seconds"
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
            except Exception as e:
                result["error"] = f"Error executing JavaScript code: {str(e)}"
        
        else:
            result["error"] = f"Code execution not supported for language: {language}"
        
        return result
    
    def format_code(self, code: str, language: str) -> str:
        """
        Format code according to language-specific style guidelines.
        
        Args:
            code: The code to format
            language: The programming language
            
        Returns:
            str: Formatted code
        """
        # Python code formatting with black
        if language == "python":
            try:
                import black
                
                # Format the code
                formatted_code = black.format_str(code, mode=black.Mode())
                return formatted_code
                
            except ImportError:
                logger.warning("black is not installed. Install it with: pip install black")
                return code
            except Exception as e:
                logger.warning(f"Error formatting Python code: {str(e)}")
                return code
        
        # JavaScript/TypeScript code formatting with prettier
        elif language in ["javascript", "typescript"]:
            try:
                import subprocess
                import tempfile
                import os
                
                # Check if prettier is installed
                try:
                    subprocess.run(["npx", "prettier", "--version"], check=True, capture_output=True)
                except (subprocess.SubprocessError, FileNotFoundError):
                    logger.warning("prettier is not installed or not in PATH")
                    return code
                
                # Create a temporary file
                with tempfile.NamedTemporaryFile(suffix=f'.{language}', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(code.encode('utf-8'))
                
                # Format the code with prettier
                process = subprocess.run(
                    ["npx", "prettier", "--write", temp_file_path],
                    capture_output=True,
                    text=True
                )
                
                # Read the formatted code
                with open(temp_file_path, 'r') as f:
                    formatted_code = f.read()
                
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
                return formatted_code
                
            except Exception as e:
                logger.warning(f"Error formatting {language} code: {str(e)}")
                return code
        
        # Default: return the original code
        return code
    
    def analyze_complexity(self, code: str, language: str) -> Dict[str, Any]:
        """
        Analyze the complexity of the code.
        
        Args:
            code: The code to analyze
            language: The programming language
            
        Returns:
            Dict[str, Any]: Complexity analysis
        """
        result = {
            "cyclomatic_complexity": None,
            "cognitive_complexity": None,
            "maintainability_index": None,
            "lines_of_code": code.count('\n') + 1,
            "comment_ratio": None
        }
        
        # Calculate comment ratio
        comment_lines = 0
        lang_patterns = self.language_patterns.get(language, {})
        comment_markers = lang_patterns.get("comment", [])
        
        if comment_markers:
            for line in code.split('\n'):
                line = line.strip()
                for marker in comment_markers:
                    if line.startswith(marker):
                        comment_lines += 1
                        break
        
        if result["lines_of_code"] > 0:
            result["comment_ratio"] = comment_lines / result["lines_of_code"]
        
        # Python complexity analysis with radon
        if language == "python":
            try:
                import radon.complexity
                import radon.metrics
                
                # Calculate cyclomatic complexity
                cc = radon.complexity.cc_visit(code)
                if cc:
                    result["cyclomatic_complexity"] = sum(c.complexity for c in cc) / len(cc)
                
                # Calculate maintainability index
                mi = radon.metrics.mi_visit(code, True)
                if mi:
                    result["maintainability_index"] = mi
                
                return result
                
            except ImportError:
                logger.warning("radon is not installed. Install it with: pip install radon")
            except Exception as e:
                logger.warning(f"Error analyzing Python code complexity: {str(e)}")
        
        # Use LLM for complexity analysis if available
        if self.llm:
            try:
                from langchain.schema import HumanMessage, SystemMessage
                
                # Create prompt for complexity analysis
                messages = [
                    SystemMessage(content=f"You are an expert {language} developer. "
                                        "Analyze the following code for complexity. "
                                        "Provide a numerical estimate (1-10) for cyclomatic complexity, "
                                        "cognitive complexity, and maintainability."),
                    HumanMessage(content=f"```{language}\n{code}\n```")
                ]
                
                # Generate analysis
                response = self.llm.generate([messages])
                analysis_text = response.generations[0][0].text.strip()
                
                # Try to extract numerical values
                cc_match = re.search(r'cyclomatic complexity[:\s]*(\d+)', analysis_text, re.IGNORECASE)
                if cc_match:
                    result["cyclomatic_complexity"] = int(cc_match.group(1))
                
                cog_match = re.search(r'cognitive complexity[:\s]*(\d+)', analysis_text, re.IGNORECASE)
                if cog_match:
                    result["cognitive_complexity"] = int(cog_match.group(1))
                
                mi_match = re.search(r'maintainability[:\s]*(\d+)', analysis_text, re.IGNORECASE)
                if mi_match:
                    result["maintainability_index"] = int(mi_match.group(1))
                
                # Add the full analysis text
                result["analysis_text"] = analysis_text
                
            except Exception as e:
                logger.warning(f"Error generating code complexity analysis: {str(e)}")
        
        return result