import logging
import os
import sys
import argparse
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import readline  # For command history
import colorama
from colorama import Fore, Style
import tempfile

# Import core components
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agent import ProblemSolverAgent
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize colorama
colorama.init(autoreset=True)

class ProblemSolverCLI:
    """Command-line interface for the AI Problem Solver"""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the CLI"""
        self.settings = settings or Settings()
        self.agent = ProblemSolverAgent(settings=self.settings)
        self.session_id = str(uuid.uuid4())
        self.solutions = []
        self.current_solution_id = None
        self.history_file = os.path.expanduser("~/.ai_problem_solver_history")
        self._load_history()
    
    def _load_history(self):
        """Load command history"""
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
                readline.set_history_length(1000)
        except Exception as e:
            logger.warning(f"Could not load history: {str(e)}")
    
    def _save_history(self):
        """Save command history"""
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            logger.warning(f"Could not save history: {str(e)}")
    
    def _print_header(self):
        """Print the CLI header"""
        print(f"{Fore.CYAN}╔═══════════════════════════════════════════════════════════════╗")
        print(f"{Fore.CYAN}║ {Fore.WHITE}{Style.BRIGHT}AI Problem Solver{Style.RESET_ALL}{Fore.CYAN} - Multi-Domain Problem Solving System ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════╝")
        print(f"{Fore.YELLOW}Type '{Fore.WHITE}help{Fore.YELLOW}' for available commands or '{Fore.WHITE}exit{Fore.YELLOW}' to quit.")
        print(f"{Fore.YELLOW}Session ID: {Fore.WHITE}{self.session_id}{Style.RESET_ALL}\n")
    
    def _print_help(self):
        """Print help information"""
        print(f"\n{Fore.CYAN}Available Commands:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}help{Style.RESET_ALL}                   - Show this help message")
        print(f"  {Fore.GREEN}exit{Style.RESET_ALL}, {Fore.GREEN}quit{Style.RESET_ALL}             - Exit the application")
        print(f"  {Fore.GREEN}clear{Style.RESET_ALL}                  - Clear the screen")
        print(f"  {Fore.GREEN}history{Style.RESET_ALL}                - Show solution history")
        print(f"  {Fore.GREEN}show <solution_num>{Style.RESET_ALL}    - Show details of a specific solution")
        print(f"  {Fore.GREEN}explain{Style.RESET_ALL}                - Explain the current solution in more detail")
        print(f"  {Fore.GREEN}verify{Style.RESET_ALL}                 - Verify the current solution")
        print(f"  {Fore.GREEN}alternatives{Style.RESET_ALL}           - Generate alternative solutions")
        print(f"  {Fore.GREEN}feedback <text>{Style.RESET_ALL}        - Provide feedback on the current solution")
        print(f"  {Fore.GREEN}refine <text>{Style.RESET_ALL}          - Refine the current solution based on feedback")
        print(f"  {Fore.GREEN}domain <domain_name>{Style.RESET_ALL}   - Specify a domain for the next problem (math, code, science, general)")
        print(f"  {Fore.GREEN}image <file_path>{Style.RESET_ALL}      - Process an image file")
        print(f"  {Fore.GREEN}voice <file_path>{Style.RESET_ALL}      - Process a voice recording file")
        print(f"  {Fore.GREEN}code <file_path>{Style.RESET_ALL}       - Process code from a file")
        print(f"  {Fore.GREEN}save <file_path>{Style.RESET_ALL}       - Save the current solution to a file")
        print(f"  {Fore.GREEN}export <file_path>{Style.RESET_ALL}     - Export the entire session to a JSON file")
        print(f"\n{Fore.CYAN}For any other input, the system will treat it as a problem to solve.{Style.RESET_ALL}\n")
    
    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _print_solution(self, solution: Dict[str, Any], detailed: bool = False):
        """Print a solution with formatting"""
        print(f"\n{Fore.CYAN}╔═══ Solution ═══{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Domain:{Style.RESET_ALL} {solution['domain']}")
        print(f"{Fore.YELLOW}Confidence:{Style.RESET_ALL} {solution.get('confidence', 0.0):.2f}\n")
        
        # Print the solution
        print(f"{Fore.GREEN}Solution:{Style.RESET_ALL}\n{solution['solution']}\n")
        
        # Print reasoning steps if available and detailed is True
        if detailed and solution.get('reasoning_steps'):
            print(f"{Fore.GREEN}Reasoning Steps:{Style.RESET_ALL}")
            for i, step in enumerate(solution['reasoning_steps'], 1):
                print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} {step}")
            print()
        
        # Print explanation if available
        if solution.get('explanation'):
            print(f"{Fore.GREEN}Explanation:{Style.RESET_ALL}\n{solution['explanation']}\n")
        
        # Print metadata if available and detailed is True
        if detailed and solution.get('metadata'):
            print(f"{Fore.GREEN}Additional Information:{Style.RESET_ALL}")
            for key, value in solution['metadata'].items():
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"{Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")
            print()
    
    def _show_history(self):
        """Show the solution history"""
        if not self.solutions:
            print(f"{Fore.YELLOW}No solutions in history.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}Solution History:{Style.RESET_ALL}")
        for i, solution in enumerate(self.solutions, 1):
            timestamp = solution.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Fore.CYAN}{i}.{Style.RESET_ALL} [{timestamp}] {Fore.YELLOW}{solution['input_type']}:{Style.RESET_ALL} {solution['input_preview']} {Fore.GREEN}({solution['result']['domain']}){Style.RESET_ALL}")
        print()
    
    def _show_solution_details(self, solution_num: int):
        """Show details of a specific solution"""
        if not self.solutions or solution_num < 1 or solution_num > len(self.solutions):
            print(f"{Fore.RED}Invalid solution number. Use 'history' to see available solutions.{Style.RESET_ALL}")
            return
        
        solution = self.solutions[solution_num - 1]
        self.current_solution_id = solution['id']
        self._print_solution(solution['result'], detailed=True)
    
    def _save_solution(self, solution: Dict[str, Any], file_path: str):
        """Save the current solution to a file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Format the solution for saving
                output = f"AI Problem Solver - Solution\n"
                output += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                output += f"Domain: {solution['domain']}\n"
                output += f"Confidence: {solution.get('confidence', 0.0):.2f}\n\n"
                
                output += f"Solution:\n{solution['solution']}\n\n"
                
                if solution.get('reasoning_steps'):
                    output += f"Reasoning Steps:\n"
                    for i, step in enumerate(solution['reasoning_steps'], 1):
                        output += f"{i}. {step}\n"
                    output += "\n"
                
                if solution.get('explanation'):
                    output += f"Explanation:\n{solution['explanation']}\n\n"
                
                f.write(output)
            
            print(f"{Fore.GREEN}Solution saved to {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving solution: {str(e)}{Style.RESET_ALL}")
    
    def _export_session(self, file_path: str):
        """Export the entire session to a JSON file"""
        try:
            export_data = {
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "solutions": self.solutions
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"{Fore.GREEN}Session exported to {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error exporting session: {str(e)}{Style.RESET_ALL}")
    
    def _process_text_input(self, text: str, domain_hint: Optional[str] = None):
        """Process text input and display the result"""
        try:
            # Process the input
            result = self.agent.process_input(
                input_text=text,
                input_type="text",
                domain_hint=domain_hint,
                session_id=self.session_id
            )
            
            # Save the solution
            solution_id = str(uuid.uuid4())
            solution = {
                "id": solution_id,
                "timestamp": datetime.now(),
                "input_type": "text",
                "input_preview": text[:50] + "..." if len(text) > 50 else text,
                "input": text,
                "result": result
            }
            self.solutions.append(solution)
            self.current_solution_id = solution_id
            
            # Display the solution
            self._print_solution(result)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing input: {str(e)}{Style.RESET_ALL}")
    
    def _process_image_input(self, file_path: str, domain_hint: Optional[str] = None):
        """Process an image file and display the result"""
        try:
            if not os.path.exists(file_path):
                print(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}Processing image...{Style.RESET_ALL}")
            
            # Process the image
            result = self.agent.process_input(
                input_image=file_path,
                input_type="image",
                domain_hint=domain_hint,
                session_id=self.session_id
            )
            
            # Save the solution
            solution_id = str(uuid.uuid4())
            solution = {
                "id": solution_id,
                "timestamp": datetime.now(),
                "input_type": "image",
                "input_preview": os.path.basename(file_path),
                "input": file_path,
                "result": result
            }
            self.solutions.append(solution)
            self.current_solution_id = solution_id
            
            # Display the solution
            self._print_solution(result)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing image: {str(e)}{Style.RESET_ALL}")
    
    def _process_voice_input(self, file_path: str, domain_hint: Optional[str] = None):
        """Process a voice recording file and display the result"""
        try:
            if not os.path.exists(file_path):
                print(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
                return
            
            print(f"{Fore.YELLOW}Processing voice recording...{Style.RESET_ALL}")
            
            # Process the voice recording
            result = self.agent.process_input(
                input_audio=file_path,
                input_type="voice",
                domain_hint=domain_hint,
                session_id=self.session_id
            )
            
            # Save the solution
            solution_id = str(uuid.uuid4())
            solution = {
                "id": solution_id,
                "timestamp": datetime.now(),
                "input_type": "voice",
                "input_preview": os.path.basename(file_path),
                "input": file_path,
                "result": result
            }
            self.solutions.append(solution)
            self.current_solution_id = solution_id
            
            # Display the solution
            self._print_solution(result)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing voice recording: {str(e)}{Style.RESET_ALL}")
    
    def _process_code_input(self, file_path: str):
        """Process code from a file and display the result"""
        try:
            if not os.path.exists(file_path):
                print(f"{Fore.RED}File not found: {file_path}{Style.RESET_ALL}")
                return
            
            # Read the code file
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            print(f"{Fore.YELLOW}Processing code...{Style.RESET_ALL}")
            
            # Process the code
            result = self.agent.process_input(
                input_text=code,
                input_type="code",
                domain_hint="code",
                session_id=self.session_id
            )
            
            # Save the solution
            solution_id = str(uuid.uuid4())
            solution = {
                "id": solution_id,
                "timestamp": datetime.now(),
                "input_type": "code",
                "input_preview": os.path.basename(file_path),
                "input": code,
                "result": result
            }
            self.solutions.append(solution)
            self.current_solution_id = solution_id
            
            # Display the solution
            self._print_solution(result)
            
        except Exception as e:
            print(f"{Fore.RED}Error processing code: {str(e)}{Style.RESET_ALL}")
    
    def _explain_solution(self):
        """Request a more detailed explanation of the current solution"""
        if not self.current_solution_id:
            print(f"{Fore.YELLOW}No current solution to explain. Solve a problem first.{Style.RESET_ALL}")
            return
        
        # Find the current solution
        current_solution = None
        for solution in self.solutions:
            if solution['id'] == self.current_solution_id:
                current_solution = solution
                break
        
        if not current_solution:
            print(f"{Fore.RED}Current solution not found.{Style.RESET_ALL}")
            return
        
        try:
            print(f"{Fore.YELLOW}Generating detailed explanation...{Style.RESET_ALL}")
            
            # Request explanation
            explanation = self.agent.explain_solution(
                session_id=self.session_id,
                solution_id=self.current_solution_id
            )
            
            # Update the solution with the explanation
            current_solution['result']['explanation'] = explanation
            
            # Display the explanation
            print(f"\n{Fore.GREEN}Detailed Explanation:{Style.RESET_ALL}\n{explanation}\n")
            
        except Exception as e:
            print(f"{Fore.RED}Error generating explanation: {str(e)}{Style.RESET_ALL}")
    
    def _verify_solution(self):
        """Verify the current solution"""
        if not self.current_solution_id:
            print(f"{Fore.YELLOW}No current solution to verify. Solve a problem first.{Style.RESET_ALL}")
            return
        
        # Find the current solution
        current_solution = None
        for solution in self.solutions:
            if solution['id'] == self.current_solution_id:
                current_solution = solution
                break
        
        if not current_solution:
            print(f"{Fore.RED}Current solution not found.{Style.RESET_ALL}")
            return
        
        try:
            print(f"{Fore.YELLOW}Verifying solution...{Style.RESET_ALL}")
            
            # Request verification
            verification_result = self.agent.verify_solution(
                session_id=self.session_id,
                solution_id=self.current_solution_id
            )
            
            # Display the verification result
            print(f"\n{Fore.CYAN}Verification Result:{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Is Correct:{Style.RESET_ALL} {verification_result['is_correct']}")
            print(f"{Fore.YELLOW}Confidence:{Style.RESET_ALL} {verification_result['confidence']:.2f}\n")
            print(f"{Fore.GREEN}Verification Details:{Style.RESET_ALL}\n{verification_result['details']}\n")
            
            # Update the solution with verification info
            if 'verification' not in current_solution['result']:
                current_solution['result']['verification'] = {}
            current_solution['result']['verification'] = verification_result
            
        except Exception as e:
            print(f"{Fore.RED}Error verifying solution: {str(e)}{Style.RESET_ALL}")
    
    def _generate_alternatives(self):
        """Generate alternative solutions for the current problem"""
        if not self.current_solution_id:
            print(f"{Fore.YELLOW}No current solution to generate alternatives for. Solve a problem first.{Style.RESET_ALL}")
            return
        
        # Find the current solution
        current_solution = None
        for solution in self.solutions:
            if solution['id'] == self.current_solution_id:
                current_solution = solution
                break
        
        if not current_solution:
            print(f"{Fore.RED}Current solution not found.{Style.RESET_ALL}")
            return
        
        try:
            print(f"{Fore.YELLOW}Generating alternative solutions...{Style.RESET_ALL}")
            
            # Request alternatives
            alternatives = self.agent.generate_alternative_solutions(
                session_id=self.session_id,
                solution_id=self.current_solution_id
            )
            
            # Update the solution with alternatives
            current_solution['result']['alternative_solutions'] = alternatives
            
            # Display the alternatives
            print(f"\n{Fore.CYAN}Alternative Solutions:{Style.RESET_ALL}")
            for i, alt in enumerate(alternatives, 1):
                print(f"\n{Fore.CYAN}Alternative {i}:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Approach:{Style.RESET_ALL} {alt['approach']}")
                print(f"{Fore.YELLOW}Confidence:{Style.RESET_ALL} {alt.get('confidence', 0.0):.2f}\n")
                print(f"{Fore.GREEN}Solution:{Style.RESET_ALL}\n{alt['solution']}\n")
            
        except Exception as e:
            print(f"{Fore.RED}Error generating alternatives: {str(e)}{Style.RESET_ALL}")
    
    def _provide_feedback(self, feedback_text: str):
        """Provide feedback on the current solution"""
        if not self.current_solution_id:
            print(f"{Fore.YELLOW}No current solution to provide feedback for. Solve a problem first.{Style.RESET_ALL}")
            return
        
        # Find the current solution
        current_solution = None
        for solution in self.solutions:
            if solution['id'] == self.current_solution_id:
                current_solution = solution
                break
        
        if not current_solution:
            print(f"{Fore.RED}Current solution not found.{Style.RESET_ALL}")
            return
        
        try:
            # Process the feedback
            self.agent.process_feedback(
                session_id=self.session_id,
                solution_id=self.current_solution_id,
                rating=None,  # No explicit rating in CLI
                feedback_text=feedback_text
            )
            
            # Add feedback to the solution
            if 'feedback' not in current_solution:
                current_solution['feedback'] = []
            
            current_solution['feedback'].append({
                "timestamp": datetime.now(),
                "text": feedback_text
            })
            
            print(f"{Fore.GREEN}Feedback recorded. Use 'refine' to improve the solution based on your feedback.{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error processing feedback: {str(e)}{Style.RESET_ALL}")
    
    def _refine_solution(self, refinement_text: str):
        """Refine the current solution based on feedback"""
        if not self.current_solution_id:
            print(f"{Fore.YELLOW}No current solution to refine. Solve a problem first.{Style.RESET_ALL}")
            return
        
        try:
            print(f"{Fore.YELLOW}Refining solution...{Style.RESET_ALL}")
            
            # Process the refinement
            result = self.agent.refine_solution(
                session_id=self.session_id,
                solution_id=self.current_solution_id,
                feedback_text=refinement_text
            )
            
            # Save the refined solution
            solution_id = str(uuid.uuid4())
            solution = {
                "id": solution_id,
                "timestamp": datetime.now(),
                "input_type": "refinement",
                "input_preview": f"Refinement of solution {self.solutions.index(next(s for s in self.solutions if s['id'] == self.current_solution_id)) + 1}",
                "input": refinement_text,
                "original_solution_id": self.current_solution_id,
                "result": result
            }
            self.solutions.append(solution)
            self.current_solution_id = solution_id
            
            # Display the refined solution
            self._print_solution(result)
            
        except Exception as e:
            print(f"{Fore.RED}Error refining solution: {str(e)}{Style.RESET_ALL}")
    
    def run(self):
        """Run the CLI interface"""
        self._clear_screen()
        self._print_header()
        
        domain_hint = None
        
        try:
            while True:
                try:
                    # Display prompt with domain if set
                    if domain_hint:
                        prompt = f"{Fore.GREEN}[{domain_hint}]{Style.RESET_ALL} > "
                    else:
                        prompt = f"{Fore.GREEN}>{Style.RESET_ALL} "
                    
                    user_input = input(prompt).strip()
                    
                    # Process commands
                    if not user_input:
                        continue
                    elif user_input.lower() in ['exit', 'quit']:
                        break
                    elif user_input.lower() == 'help':
                        self._print_help()
                    elif user_input.lower() == 'clear':
                        self._clear_screen()
                        self._print_header()
                    elif user_input.lower() == 'history':
                        self._show_history()
                    elif user_input.lower().startswith('show '):
                        try:
                            solution_num = int(user_input[5:].strip())
                            self._show_solution_details(solution_num)
                        except ValueError:
                            print(f"{Fore.RED}Invalid solution number. Use 'history' to see available solutions.{Style.RESET_ALL}")
                    elif user_input.lower() == 'explain':
                        self._explain_solution()
                    elif user_input.lower() == 'verify':
                        self._verify_solution()
                    elif user_input.lower() == 'alternatives':
                        self._generate_alternatives()
                    elif user_input.lower().startswith('feedback '):
                        feedback_text = user_input[9:].strip()
                        self._provide_feedback(feedback_text)
                    elif user_input.lower().startswith('refine '):
                        refinement_text = user_input[7:].strip()
                        self._refine_solution(refinement_text)
                    elif user_input.lower().startswith('domain '):
                        domain = user_input[7:].strip().lower()
                        if domain in ['math', 'code', 'science', 'general']:
                            domain_hint = domain
                            print(f"{Fore.GREEN}Domain set to {domain_hint}.{Style.RESET_ALL}")
                        else:
                            print(f"{Fore.RED}Invalid domain. Use 'math', 'code', 'science', or 'general'.{Style.RESET_ALL}")
                    elif user_input.lower().startswith('image '):
                        file_path = user_input[6:].strip()
                        self._process_image_input(file_path, domain_hint)
                    elif user_input.lower().startswith('voice '):
                        file_path = user_input[6:].strip()
                        self._process_voice_input(file_path, domain_hint)
                    elif user_input.lower().startswith('code '):
                        file_path = user_input[5:].strip()
                        self._process_code_input(file_path)
                    elif user_input.lower().startswith('save '):
                        if not self.current_solution_id:
                            print(f"{Fore.YELLOW}No current solution to save. Solve a problem first.{Style.RESET_ALL}")
                            continue
                        
                        file_path = user_input[5:].strip()
                        current_solution = next((s for s in self.solutions if s['id'] == self.current_solution_id), None)
                        if current_solution:
                            self._save_solution(current_solution['result'], file_path)
                    elif user_input.lower().startswith('export '):
                        file_path = user_input[7:].strip()
                        self._export_session(file_path)
                    else:
                        # Treat as a problem to solve
                        self._process_text_input(user_input, domain_hint)
                
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit.")
                except Exception as e:
                    print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        
        finally:
            # Save command history
            self._save_history()
            print(f"\n{Fore.CYAN}Thank you for using AI Problem Solver!{Style.RESET_ALL}")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="AI Problem Solver CLI")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--problem', type=str, help='Problem to solve')
    parser.add_argument('--domain', type=str, choices=['math', 'code', 'science', 'general'], 
                        help='Problem domain')
    parser.add_argument('--image', type=str, help='Path to image file to process')
    parser.add_argument('--voice', type=str, help='Path to voice recording file to process')
    parser.add_argument('--code', type=str, help='Path to code file to process')
    parser.add_argument('--output', type=str, help='Path to save the solution')
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_arguments()
    
    # Load settings
    settings = Settings()
    if args.config:
        settings.load_from_file(args.config)
    
    # Create CLI
    cli = ProblemSolverCLI(settings=settings)
    
    # Process single command if provided
    if args.problem or args.image or args.voice or args.code:
        if args.problem:
            cli._process_text_input(args.problem, args.domain)
        elif args.image:
            cli._process_image_input(args.image, args.domain)
        elif args.voice:
            cli._process_voice_input(args.voice, args.domain)
        elif args.code:
            cli._process_code_input(args.code)
        
        # Save output if requested
        if args.output and cli.current_solution_id:
            current_solution = next((s for s in cli.solutions if s['id'] == cli.current_solution_id), None)
            if current_solution:
                cli._save_solution(current_solution['result'], args.output)
    else:
        # Run interactive CLI
        cli.run()

class CommandLineInterface:
    """Wrapper class for ProblemSolverCLI to maintain compatibility with main.py"""
    
    def __init__(self, agent=None, settings=None):
        """Initialize the CLI interface
        
        Args:
            agent: ProblemSolverAgent instance (optional)
            settings: Settings instance (optional)
        """
        self.settings = settings
        self.agent = agent
        self.cli = None
    
    def start(self):
        """Start the command-line interface"""
        # Create a new CLI instance
        # If agent was provided in constructor, use it, otherwise let ProblemSolverCLI create one
        if self.agent:
            # Create a CLI with the provided agent
            self.cli = ProblemSolverCLI(settings=self.settings)
            self.cli.agent = self.agent
        else:
            # Let ProblemSolverCLI create its own agent
            self.cli = ProblemSolverCLI(settings=self.settings)
        
        # Run the CLI
        self.cli.run()

if __name__ == "__main__":
    main()