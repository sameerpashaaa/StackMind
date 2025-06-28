import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import sympy as sp
from sympy import symbols, solve, Eq, simplify, diff, integrate, Matrix
from sympy.parsing.sympy_parser import parse_expr

# Remove direct import of ChatOpenAI
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class ScienceSolver:
    """
    A domain-specific solver for scientific problems.
    
    This solver can handle problems in:
    1. Physics (mechanics, thermodynamics, electromagnetism, etc.)
    2. Chemistry (stoichiometry, reactions, equilibrium, etc.)
    3. Biology (genetics, ecology, etc.)
    4. Earth Science (geology, meteorology, etc.)
    5. Astronomy (celestial mechanics, stellar evolution, etc.)
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """
        Initialize the science solver.
        Args:
            llm: Language model for science problem solving (any LangChain-compatible chat model)
        """
        self.llm = llm
        # Constants for physics calculations
        self.constants = {
            'G': 6.67430e-11,  # Gravitational constant (m^3 kg^-1 s^-2)
            'c': 299792458,    # Speed of light in vacuum (m/s)
            'h': 6.62607015e-34,  # Planck constant (J⋅s)
            'e': 1.602176634e-19,  # Elementary charge (C)
            'k_B': 1.380649e-23,  # Boltzmann constant (J/K)
            'N_A': 6.02214076e23,  # Avogadro constant (mol^-1)
            'R': 8.31446261815324,  # Gas constant (J/(mol⋅K))
            'sigma': 5.670374419e-8,  # Stefan-Boltzmann constant (W/(m^2⋅K^4))
            'epsilon_0': 8.8541878128e-12,  # Vacuum permittivity (F/m)
            'mu_0': 1.25663706212e-6,  # Vacuum permeability (H/m)
            'g': 9.80665,  # Standard acceleration due to gravity (m/s^2)
            'atm': 101325,  # Standard atmospheric pressure (Pa)
        }
        
        # Periodic table data for chemistry calculations
        self.periodic_table = {
            'H': {'atomic_number': 1, 'atomic_mass': 1.008, 'name': 'Hydrogen'},
            'He': {'atomic_number': 2, 'atomic_mass': 4.0026, 'name': 'Helium'},
            'Li': {'atomic_number': 3, 'atomic_mass': 6.94, 'name': 'Lithium'},
            'Be': {'atomic_number': 4, 'atomic_mass': 9.0122, 'name': 'Beryllium'},
            'B': {'atomic_number': 5, 'atomic_mass': 10.81, 'name': 'Boron'},
            'C': {'atomic_number': 6, 'atomic_mass': 12.011, 'name': 'Carbon'},
            'N': {'atomic_number': 7, 'atomic_mass': 14.007, 'name': 'Nitrogen'},
            'O': {'atomic_number': 8, 'atomic_mass': 15.999, 'name': 'Oxygen'},
            'F': {'atomic_number': 9, 'atomic_mass': 18.998, 'name': 'Fluorine'},
            'Ne': {'atomic_number': 10, 'atomic_mass': 20.180, 'name': 'Neon'},
            'Na': {'atomic_number': 11, 'atomic_mass': 22.990, 'name': 'Sodium'},
            'Mg': {'atomic_number': 12, 'atomic_mass': 24.305, 'name': 'Magnesium'},
            'Al': {'atomic_number': 13, 'atomic_mass': 26.982, 'name': 'Aluminum'},
            'Si': {'atomic_number': 14, 'atomic_mass': 28.085, 'name': 'Silicon'},
            'P': {'atomic_number': 15, 'atomic_mass': 30.974, 'name': 'Phosphorus'},
            'S': {'atomic_number': 16, 'atomic_mass': 32.06, 'name': 'Sulfur'},
            'Cl': {'atomic_number': 17, 'atomic_mass': 35.45, 'name': 'Chlorine'},
            'Ar': {'atomic_number': 18, 'atomic_mass': 39.948, 'name': 'Argon'},
            'K': {'atomic_number': 19, 'atomic_mass': 39.098, 'name': 'Potassium'},
            'Ca': {'atomic_number': 20, 'atomic_mass': 40.078, 'name': 'Calcium'},
            # Add more elements as needed
        }
    
    def solve(self, problem: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Solve a scientific problem.
        
        Args:
            problem: The problem description
            context: Additional context for the problem (optional)
            
        Returns:
            Dict containing the solution, explanation, and other relevant information
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'domain': None,
            'subdomain': None,
            'equations': [],
            'error': None
        }
        
        try:
            # Detect the scientific domain
            domain, subdomain = self._detect_domain(problem)
            result['domain'] = domain
            result['subdomain'] = subdomain
            
            # Solve based on domain
            if domain == 'physics':
                solution = self._solve_physics_problem(problem, subdomain, context)
            elif domain == 'chemistry':
                solution = self._solve_chemistry_problem(problem, subdomain, context)
            elif domain == 'biology':
                solution = self._solve_biology_problem(problem, subdomain, context)
            elif domain == 'earth_science':
                solution = self._solve_earth_science_problem(problem, subdomain, context)
            elif domain == 'astronomy':
                solution = self._solve_astronomy_problem(problem, subdomain, context)
            else:
                # Default to general science solution
                solution = self._solve_with_llm(problem, domain, subdomain, context)
            
            # Update result with solution
            result.update(solution)
            
        except Exception as e:
            logger.error(f"Error in science solver: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _detect_domain(self, problem: str) -> Tuple[str, Optional[str]]:
        """
        Detect the scientific domain and subdomain of the problem.
        
        Args:
            problem: The problem description
            
        Returns:
            Tuple of (domain, subdomain)
        """
        problem_lower = problem.lower()
        
        # Physics keywords
        physics_keywords = {
            'mechanics': ['force', 'motion', 'velocity', 'acceleration', 'momentum', 'newton', 'kinetic', 'potential', 'energy', 'joule', 'watt', 'friction', 'gravity', 'projectile', 'trajectory'],
            'thermodynamics': ['heat', 'temperature', 'entropy', 'thermal', 'kelvin', 'celsius', 'fahrenheit', 'pressure', 'volume', 'gas', 'expansion', 'contraction', 'thermodynamic', 'adiabatic', 'isothermal'],
            'electromagnetism': ['electric', 'magnetic', 'field', 'charge', 'current', 'voltage', 'resistance', 'capacitance', 'inductance', 'circuit', 'ohm', 'ampere', 'volt', 'coulomb', 'tesla', 'gauss', 'maxwell', 'faraday'],
            'optics': ['light', 'reflection', 'refraction', 'diffraction', 'interference', 'polarization', 'lens', 'mirror', 'prism', 'wavelength', 'frequency', 'photon', 'laser', 'optical'],
            'quantum': ['quantum', 'wave function', 'uncertainty', 'heisenberg', 'schrödinger', 'planck', 'bohr', 'atomic', 'subatomic', 'particle', 'quark', 'lepton', 'hadron', 'fermion', 'boson'],
            'relativity': ['relativity', 'einstein', 'spacetime', 'lorentz', 'invariant', 'relativistic', 'time dilation', 'length contraction', 'reference frame', 'inertial', 'gravitational']
        }
        
        # Chemistry keywords
        chemistry_keywords = {
            'stoichiometry': ['mole', 'stoichiometry', 'reaction', 'reactant', 'product', 'yield', 'limiting reagent', 'excess reagent', 'theoretical yield', 'actual yield', 'percent yield', 'molecular weight', 'atomic weight', 'formula weight'],
            'thermochemistry': ['enthalpy', 'entropy', 'gibbs', 'free energy', 'exothermic', 'endothermic', 'heat of reaction', 'heat of formation', 'calorimetry', 'bond energy', 'hess law'],
            'equilibrium': ['equilibrium', 'reversible', 'le chatelier', 'equilibrium constant', 'reaction quotient', 'dynamic equilibrium', 'chemical equilibrium', 'solubility product', 'common ion effect'],
            'kinetics': ['rate', 'kinetics', 'reaction rate', 'rate constant', 'activation energy', 'catalyst', 'inhibitor', 'order of reaction', 'half-life', 'arrhenius', 'mechanism', 'elementary step'],
            'acid_base': ['acid', 'base', 'ph', 'poh', 'buffer', 'titration', 'neutralization', 'hydronium', 'hydroxide', 'conjugate', 'bronsted', 'lewis', 'amphoteric', 'dissociation', 'ionization'],
            'organic': ['organic', 'carbon', 'hydrocarbon', 'alkane', 'alkene', 'alkyne', 'aromatic', 'functional group', 'isomer', 'polymer', 'monomer', 'substitution', 'elimination', 'addition']
        }
        
        # Biology keywords
        biology_keywords = {
            'genetics': ['gene', 'dna', 'rna', 'chromosome', 'allele', 'genotype', 'phenotype', 'heredity', 'inheritance', 'mutation', 'recombination', 'dominant', 'recessive', 'punnett square', 'mendel'],
            'ecology': ['ecosystem', 'habitat', 'niche', 'population', 'community', 'biome', 'food chain', 'food web', 'trophic level', 'producer', 'consumer', 'decomposer', 'biodiversity', 'succession'],
            'cell_biology': ['cell', 'membrane', 'organelle', 'nucleus', 'mitochondria', 'chloroplast', 'ribosome', 'golgi', 'endoplasmic', 'lysosome', 'cytoplasm', 'cytoskeleton', 'transport'],
            'physiology': ['organ', 'tissue', 'system', 'homeostasis', 'hormone', 'nerve', 'muscle', 'blood', 'heart', 'lung', 'kidney', 'liver', 'brain', 'digestion', 'respiration', 'circulation'],
            'evolution': ['evolution', 'natural selection', 'adaptation', 'fitness', 'speciation', 'phylogeny', 'taxonomy', 'cladistics', 'homology', 'analogy', 'convergent', 'divergent', 'darwin', 'fossil']
        }
        
        # Earth Science keywords
        earth_science_keywords = {
            'geology': ['rock', 'mineral', 'plate tectonics', 'earthquake', 'volcano', 'erosion', 'weathering', 'sediment', 'strata', 'fossil', 'geologic time', 'mountain', 'fault', 'glacier'],
            'meteorology': ['weather', 'climate', 'atmosphere', 'precipitation', 'humidity', 'pressure', 'front', 'cyclone', 'anticyclone', 'hurricane', 'tornado', 'wind', 'cloud', 'storm'],
            'oceanography': ['ocean', 'sea', 'tide', 'current', 'wave', 'salinity', 'marine', 'coastal', 'reef', 'abyssal', 'benthic', 'pelagic', 'upwelling', 'downwelling'],
            'hydrology': ['water', 'river', 'lake', 'stream', 'groundwater', 'aquifer', 'watershed', 'drainage', 'runoff', 'infiltration', 'precipitation', 'evaporation', 'transpiration', 'water cycle']
        }
        
        # Astronomy keywords
        astronomy_keywords = {
            'celestial_mechanics': ['orbit', 'gravity', 'kepler', 'newton', 'planetary motion', 'elliptical', 'perihelion', 'aphelion', 'conjunction', 'opposition', 'transit', 'eclipse'],
            'stellar_evolution': ['star', 'stellar', 'main sequence', 'red giant', 'white dwarf', 'neutron star', 'black hole', 'supernova', 'nova', 'nebula', 'protostar', 'hertzsprung-russell', 'luminosity', 'magnitude'],
            'cosmology': ['universe', 'big bang', 'cosmic', 'expansion', 'dark matter', 'dark energy', 'cosmic microwave background', 'redshift', 'blueshift', 'hubble', 'galaxy', 'quasar', 'pulsar', 'inflation'],
            'planetary_science': ['planet', 'moon', 'asteroid', 'comet', 'meteor', 'solar system', 'terrestrial', 'jovian', 'ring', 'atmosphere', 'crater', 'impact', 'rotation', 'revolution']
        }
        
        # Combine all domains and their subdomains
        all_domains = {
            'physics': physics_keywords,
            'chemistry': chemistry_keywords,
            'biology': biology_keywords,
            'earth_science': earth_science_keywords,
            'astronomy': astronomy_keywords
        }
        
        # Count keyword matches for each domain and subdomain
        domain_scores = {domain: 0 for domain in all_domains}
        subdomain_scores = {}
        
        for domain, subdomains in all_domains.items():
            for subdomain, keywords in subdomains.items():
                score = sum(1 for keyword in keywords if keyword in problem_lower)
                if score > 0:
                    domain_scores[domain] += score
                    subdomain_scores[(domain, subdomain)] = score
        
        # Find the domain with the highest score
        max_domain_score = max(domain_scores.values())
        if max_domain_score == 0:
            return 'general_science', None
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
        
        # Find the subdomain with the highest score within the best domain
        best_subdomain = None
        max_subdomain_score = 0
        
        for (domain, subdomain), score in subdomain_scores.items():
            if domain == best_domain and score > max_subdomain_score:
                max_subdomain_score = score
                best_subdomain = subdomain
        
        return best_domain, best_subdomain
    
    def _solve_physics_problem(self, problem: str, subdomain: Optional[str], 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a physics problem.
        
        Args:
            problem: The problem description
            subdomain: The physics subdomain (mechanics, thermodynamics, etc.)
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'equations': []
        }
        
        try:
            # Extract relevant quantities and units from the problem
            quantities = self._extract_quantities(problem)
            result['quantities'] = quantities
            
            # Extract equations from the problem
            equations = self._extract_equations(problem)
            result['equations'] = equations
            
            # Solve based on subdomain
            if subdomain == 'mechanics':
                solution = self._solve_mechanics_problem(problem, quantities, equations, context)
            elif subdomain == 'thermodynamics':
                solution = self._solve_thermodynamics_problem(problem, quantities, equations, context)
            elif subdomain == 'electromagnetism':
                solution = self._solve_electromagnetism_problem(problem, quantities, equations, context)
            elif subdomain == 'optics':
                solution = self._solve_optics_problem(problem, quantities, equations, context)
            elif subdomain == 'quantum':
                solution = self._solve_quantum_problem(problem, quantities, equations, context)
            elif subdomain == 'relativity':
                solution = self._solve_relativity_problem(problem, quantities, equations, context)
            else:
                # Default to general physics solution
                solution = self._solve_with_llm(problem, 'physics', subdomain, context)
            
            # Update result with solution
            result.update(solution)
            
        except Exception as e:
            logger.error(f"Error solving physics problem: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _solve_chemistry_problem(self, problem: str, subdomain: Optional[str], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a chemistry problem.
        
        Args:
            problem: The problem description
            subdomain: The chemistry subdomain (stoichiometry, thermochemistry, etc.)
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'equations': []
        }
        
        try:
            # Extract chemical equations and reactions
            chemical_equations = self._extract_chemical_equations(problem)
            result['chemical_equations'] = chemical_equations
            
            # Extract quantities and units
            quantities = self._extract_quantities(problem)
            result['quantities'] = quantities
            
            # Solve based on subdomain
            if subdomain == 'stoichiometry':
                solution = self._solve_stoichiometry_problem(problem, chemical_equations, quantities, context)
            elif subdomain == 'thermochemistry':
                solution = self._solve_thermochemistry_problem(problem, chemical_equations, quantities, context)
            elif subdomain == 'equilibrium':
                solution = self._solve_equilibrium_problem(problem, chemical_equations, quantities, context)
            elif subdomain == 'kinetics':
                solution = self._solve_kinetics_problem(problem, chemical_equations, quantities, context)
            elif subdomain == 'acid_base':
                solution = self._solve_acid_base_problem(problem, chemical_equations, quantities, context)
            elif subdomain == 'organic':
                solution = self._solve_organic_chemistry_problem(problem, chemical_equations, quantities, context)
            else:
                # Default to general chemistry solution
                solution = self._solve_with_llm(problem, 'chemistry', subdomain, context)
            
            # Update result with solution
            result.update(solution)
            
        except Exception as e:
            logger.error(f"Error solving chemistry problem: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _solve_biology_problem(self, problem: str, subdomain: Optional[str], 
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a biology problem.
        
        Args:
            problem: The problem description
            subdomain: The biology subdomain (genetics, ecology, etc.)
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        # For biology problems, we'll primarily use the LLM
        return self._solve_with_llm(problem, 'biology', subdomain, context)
    
    def _solve_earth_science_problem(self, problem: str, subdomain: Optional[str], 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an earth science problem.
        
        Args:
            problem: The problem description
            subdomain: The earth science subdomain (geology, meteorology, etc.)
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        # For earth science problems, we'll primarily use the LLM
        return self._solve_with_llm(problem, 'earth_science', subdomain, context)
    
    def _solve_astronomy_problem(self, problem: str, subdomain: Optional[str], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an astronomy problem.
        
        Args:
            problem: The problem description
            subdomain: The astronomy subdomain (celestial mechanics, stellar evolution, etc.)
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': [],
            'equations': []
        }
        
        try:
            # Extract relevant quantities and units
            quantities = self._extract_quantities(problem)
            result['quantities'] = quantities
            
            # Extract equations
            equations = self._extract_equations(problem)
            result['equations'] = equations
            
            # For astronomy problems, we'll primarily use the LLM
            # but we could add specialized solvers for celestial mechanics, etc.
            solution = self._solve_with_llm(problem, 'astronomy', subdomain, context)
            
            # Update result with solution
            result.update(solution)
            
        except Exception as e:
            logger.error(f"Error solving astronomy problem: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _extract_quantities(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Extract numerical quantities and their units from text.
        
        Args:
            text: The text to extract quantities from
            
        Returns:
            Dict of quantities with their values and units
        """
        quantities = {}
        
        # Regular expression to match numbers with optional units
        # This pattern matches numbers like 5, 5.2, 5,200, 5.2e3, etc. with optional units
        pattern = r'(\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?)\s*([a-zA-Z°/%]+)?'
        
        matches = re.finditer(pattern, text)
        
        for i, match in enumerate(matches):
            value_str, unit = match.groups()
            
            # Clean up the value string
            value_str = value_str.replace(',', '')
            
            try:
                value = float(value_str)
            except ValueError:
                continue
            
            # Generate a default name based on position
            name = f"quantity_{i+1}"
            
            # Try to infer the name from surrounding text
            context_start = max(0, match.start() - 50)
            context_end = min(len(text), match.end() + 50)
            context_text = text[context_start:context_end]
            
            # Look for potential variable names before the quantity
            name_match = re.search(r'([a-zA-Z][a-zA-Z0-9_]*)[\s=:]*$', text[context_start:match.start()])
            if name_match:
                name = name_match.group(1).strip()
            
            quantities[name] = {
                'value': value,
                'unit': unit if unit else None,
                'position': match.start(),
                'context': context_text
            }
        
        return quantities
    
    def _extract_equations(self, text: str) -> List[str]:
        """
        Extract mathematical equations from text.
        
        Args:
            text: The text to extract equations from
            
        Returns:
            List of equation strings
        """
        equations = []
        
        # Look for equation patterns
        # This is a simple pattern that looks for strings containing =, +, -, *, /, ^, etc.
        equation_pattern = r'[a-zA-Z0-9_\s\+\-\*\/\^\(\)=<>≤≥±∓×÷√∛∜∑∏∫∬∭∮∯∰∱∲∳]+[=<>≤≥]'
        
        # Also look for equations in LaTeX format
        latex_pattern = r'\\begin\{equation\}.*?\\end\{equation\}|\$\$.*?\$\$|\$.*?\$'
        
        # Find all matches
        equation_matches = re.finditer(equation_pattern, text)
        latex_matches = re.finditer(latex_pattern, text, re.DOTALL)
        
        # Extract equations
        for match in equation_matches:
            equation = match.group(0).strip()
            if len(equation) > 3 and '=' in equation:  # Ensure it's a meaningful equation
                equations.append(equation)
        
        for match in latex_matches:
            latex_eq = match.group(0).strip()
            equations.append(latex_eq)
        
        return equations
    
    def _extract_chemical_equations(self, text: str) -> List[str]:
        """
        Extract chemical equations and reactions from text.
        
        Args:
            text: The text to extract chemical equations from
            
        Returns:
            List of chemical equation strings
        """
        chemical_equations = []
        
        # Pattern for chemical equations (e.g., "2H2 + O2 -> 2H2O")
        patterns = [
            r'[A-Z][a-z]?\d*(?:\([^)]+\)\d*)?(?:\s*[+]\s*[A-Z][a-z]?\d*(?:\([^)]+\)\d*)?)*\s*(?:->|→|=|⟶|⇌|⇋|⇄|⇆|⇔)\s*[A-Z][a-z]?\d*(?:\([^)]+\)\d*)?(?:\s*[+]\s*[A-Z][a-z]?\d*(?:\([^)]+\)\d*)?)*',
            r'[A-Z][a-z]?(?:\d+)?(?:\([^)]+\)(?:\d+)?)?(?:\s*[+]\s*[A-Z][a-z]?(?:\d+)?(?:\([^)]+\)(?:\d+)?)?)*\s*(?:->|→|=|⟶|⇌|⇋|⇄|⇆|⇔)\s*[A-Z][a-z]?(?:\d+)?(?:\([^)]+\)(?:\d+)?)?(?:\s*[+]\s*[A-Z][a-z]?(?:\d+)?(?:\([^)]+\)(?:\d+)?)?)*'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                equation = match.group(0).strip()
                chemical_equations.append(equation)
        
        return chemical_equations
    
    def _solve_mechanics_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                               equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a mechanics problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': []
        }
        
        try:
            # Check if we have enough information to solve analytically
            if equations and quantities:
                # Try to solve using symbolic math
                symbolic_solution = self._solve_equations_symbolically(equations, quantities)
                if symbolic_solution.get('solution'):
                    return symbolic_solution
            
            # If we can't solve analytically, use the LLM
            return self._solve_with_llm(problem, 'physics', 'mechanics', context)
            
        except Exception as e:
            logger.error(f"Error solving mechanics problem: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _solve_thermodynamics_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                                     equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a thermodynamics problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        # Similar to mechanics, try symbolic solution first, then fall back to LLM
        return self._solve_with_llm(problem, 'physics', 'thermodynamics', context)
    
    def _solve_electromagnetism_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                                       equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an electromagnetism problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        # Similar approach as other physics subdomains
        return self._solve_with_llm(problem, 'physics', 'electromagnetism', context)
    
    def _solve_optics_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                            equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an optics problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'physics', 'optics', context)
    
    def _solve_quantum_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                             equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a quantum physics problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'physics', 'quantum', context)
    
    def _solve_relativity_problem(self, problem: str, quantities: Dict[str, Dict[str, Any]], 
                                equations: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a relativity problem.
        
        Args:
            problem: The problem description
            quantities: Extracted quantities
            equations: Extracted equations
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'physics', 'relativity', context)
    
    def _solve_stoichiometry_problem(self, problem: str, chemical_equations: List[str], 
                                   quantities: Dict[str, Dict[str, Any]], 
                                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a stoichiometry problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'stoichiometry', context)
    
    def _solve_thermochemistry_problem(self, problem: str, chemical_equations: List[str], 
                                      quantities: Dict[str, Dict[str, Any]], 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a thermochemistry problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'thermochemistry', context)
    
    def _solve_equilibrium_problem(self, problem: str, chemical_equations: List[str], 
                                 quantities: Dict[str, Dict[str, Any]], 
                                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a chemical equilibrium problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'equilibrium', context)
    
    def _solve_kinetics_problem(self, problem: str, chemical_equations: List[str], 
                              quantities: Dict[str, Dict[str, Any]], 
                              context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a chemical kinetics problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'kinetics', context)
    
    def _solve_acid_base_problem(self, problem: str, chemical_equations: List[str], 
                               quantities: Dict[str, Dict[str, Any]], 
                               context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an acid-base problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'acid_base', context)
    
    def _solve_organic_chemistry_problem(self, problem: str, chemical_equations: List[str], 
                                       quantities: Dict[str, Dict[str, Any]], 
                                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve an organic chemistry problem.
        
        Args:
            problem: The problem description
            chemical_equations: Extracted chemical equations
            quantities: Extracted quantities
            context: Additional context
            
        Returns:
            Dict containing the solution and explanation
        """
        return self._solve_with_llm(problem, 'chemistry', 'organic', context)
    
    def _solve_equations_symbolically(self, equations: List[str], 
                                    quantities: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Solve equations symbolically using SymPy.
        
        Args:
            equations: List of equation strings
            quantities: Dict of quantities with their values
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': []
        }
        
        try:
            if not equations:
                return result
            
            # Convert equations to SymPy expressions
            sympy_equations = []
            variables = set()
            
            for eq_str in equations:
                # Check if it's an equation (contains =)
                if '=' in eq_str:
                    left_str, right_str = eq_str.split('=', 1)
                    left_expr = parse_expr(left_str.strip())
                    right_expr = parse_expr(right_str.strip())
                    sympy_eq = Eq(left_expr, right_expr)
                    sympy_equations.append(sympy_eq)
                    
                    # Collect variables
                    variables.update(left_expr.free_symbols)
                    variables.update(right_expr.free_symbols)
            
            if not sympy_equations or not variables:
                return result
            
            # Substitute known quantities
            substitutions = {}
            for var_name, var_data in quantities.items():
                # Try to match variable name with a symbol in the equations
                for symbol in variables:
                    if str(symbol) == var_name:
                        substitutions[symbol] = var_data['value']
                        break
            
            # Apply substitutions to equations
            substituted_equations = [eq.subs(substitutions) for eq in sympy_equations]
            
            # Determine which variables are unknown (not in substitutions)
            unknowns = [var for var in variables if var not in substitutions]
            
            if not unknowns:
                # All variables are known, just verify the equations
                result['solution'] = "All variables are known. Equations are satisfied."
                result['explanation'] = "The provided equations are satisfied with the given values."
                return result
            
            # Solve for the unknowns
            solution = solve(substituted_equations, unknowns)
            
            if solution:
                # Format the solution
                if isinstance(solution, dict):
                    solution_str = ", ".join([f"{var} = {val}" for var, val in solution.items()])
                elif isinstance(solution, list):
                    if all(isinstance(sol, dict) for sol in solution):
                        solution_str = "\n".join([f"Solution {i+1}: " + ", ".join([f"{var} = {val}" for var, val in sol.items()]) for i, sol in enumerate(solution)])
                    else:
                        solution_str = str(solution)
                else:
                    solution_str = str(solution)
                
                result['solution'] = solution_str
                result['explanation'] = f"Solved the system of equations symbolically. Found values for {', '.join([str(var) for var in unknowns])}."
                result['steps'] = [f"Started with equations: {', '.join([str(eq) for eq in sympy_equations])}",
                                 f"Substituted known values: {substitutions}",
                                 f"Solved for unknowns: {', '.join([str(var) for var in unknowns])}",
                                 f"Solution: {solution_str}"]
            else:
                result['solution'] = "Could not find a solution to the system of equations."
                result['explanation'] = "The system of equations may be inconsistent or have no solution."
            
        except Exception as e:
            logger.error(f"Error solving equations symbolically: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def _solve_with_llm(self, problem: str, domain: str, subdomain: Optional[str], 
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve a problem using the language model.
        
        Args:
            problem: The problem description
            domain: The scientific domain
            subdomain: The specific subdomain (optional)
            context: Additional context (optional)
            
        Returns:
            Dict containing the solution and explanation
        """
        result = {
            'solution': None,
            'explanation': None,
            'steps': []
        }
        
        try:
            # Use LLM to solve the problem
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create domain-specific prompt
            domain_prompts = {
                'physics': "You are an expert physicist specializing in solving physics problems. ",
                'chemistry': "You are an expert chemist specializing in solving chemistry problems. ",
                'biology': "You are an expert biologist specializing in solving biology problems. ",
                'earth_science': "You are an expert in earth sciences specializing in solving geology, meteorology, and related problems. ",
                'astronomy': "You are an expert astronomer specializing in solving astronomy and astrophysics problems. ",
                'general_science': "You are an expert scientist with broad knowledge across multiple scientific disciplines. "
            }
            
            subdomain_prompts = {
                'mechanics': "You have particular expertise in classical mechanics, forces, motion, and energy. ",
                'thermodynamics': "You have particular expertise in thermodynamics, heat transfer, and thermal processes. ",
                'electromagnetism': "You have particular expertise in electromagnetism, electric fields, magnetic fields, and circuits. ",
                'optics': "You have particular expertise in optics, light propagation, and optical phenomena. ",
                'quantum': "You have particular expertise in quantum mechanics and quantum phenomena. ",
                'relativity': "You have particular expertise in special and general relativity. ",
                'stoichiometry': "You have particular expertise in chemical stoichiometry and reaction calculations. ",
                'thermochemistry': "You have particular expertise in thermochemistry and energy changes in chemical reactions. ",
                'equilibrium': "You have particular expertise in chemical equilibrium and equilibrium calculations. ",
                'kinetics': "You have particular expertise in chemical kinetics and reaction rates. ",
                'acid_base': "You have particular expertise in acid-base chemistry and pH calculations. ",
                'organic': "You have particular expertise in organic chemistry and carbon compounds. ",
                'genetics': "You have particular expertise in genetics and heredity. ",
                'ecology': "You have particular expertise in ecology and ecosystems. ",
                'cell_biology': "You have particular expertise in cell biology and cellular processes. ",
                'physiology': "You have particular expertise in physiology and organ systems. ",
                'evolution': "You have particular expertise in evolutionary biology. ",
                'geology': "You have particular expertise in geology and Earth's structure. ",
                'meteorology': "You have particular expertise in meteorology and atmospheric science. ",
                'oceanography': "You have particular expertise in oceanography and marine science. ",
                'hydrology': "You have particular expertise in hydrology and water systems. ",
                'celestial_mechanics': "You have particular expertise in celestial mechanics and orbital dynamics. ",
                'stellar_evolution': "You have particular expertise in stellar evolution and stellar processes. ",
                'cosmology': "You have particular expertise in cosmology and the large-scale universe. ",
                'planetary_science': "You have particular expertise in planetary science and solar system objects. "
            }
            
            # Build the system prompt
            system_content = domain_prompts.get(domain, domain_prompts['general_science'])
            if subdomain and subdomain in subdomain_prompts:
                system_content += subdomain_prompts[subdomain]
            
            system_content += "Solve the following problem step by step, showing all your work. "
            system_content += "Include relevant equations, calculations, and explanations. "
            system_content += "Provide a clear final answer."
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"Problem: {problem}")
            ]
            
            # Add context if available
            if context:
                context_str = "\n\nAdditional context:\n"
                for key, value in context.items():
                    context_str += f"{key}: {value}\n"
                messages.append(HumanMessage(content=context_str))
            
            # Generate solution
            response = self.llm.generate([messages])
            solution_text = response.generations[0][0].text.strip()
            
            # Parse the solution
            # Extract steps and final answer
            steps = []
            explanation = ""
            solution = None
            
            # Simple parsing - in a real implementation, you would use more sophisticated parsing
            lines = solution_text.split('\n')
            for line in lines:
                if line.strip():
                    steps.append(line.strip())
            
            # Try to extract the final answer
            answer_patterns = [
                r'final answer:?\s*(.+)',
                r'answer:?\s*(.+)',
                r'solution:?\s*(.+)',
                r'result:?\s*(.+)',
                r'therefore,?\s*(.+)'
            ]
            
            for pattern in answer_patterns:
                for line in reversed(lines):  # Start from the end
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        solution = match.group(1).strip()
                        break
                if solution:
                    break
            
            # If no clear answer found, use the last line
            if not solution and steps:
                solution = steps[-1]
            
            # Generate explanation
            explanation = "Solution generated using scientific principles and problem-solving techniques."
            
            result['solution'] = solution
            result['steps'] = steps
            result['explanation'] = explanation
            
        except Exception as e:
            logger.error(f"Error solving with LLM: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def verify_solution(self, problem: str, solution: Any, domain: str, 
                       subdomain: Optional[str] = None) -> Dict[str, bool]:
        """
        Verify if a solution is correct.
        
        Args:
            problem: The problem description
            solution: The solution to verify
            domain: The scientific domain
            subdomain: The specific subdomain (optional)
            
        Returns:
            Dict containing verification results
        """
        result = {
            'is_correct': False,
            'confidence': 0.0,
            'explanation': None,
            'error': None
        }
        
        try:
            # Use LLM to verify the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Create prompt for verification
            messages = [
                SystemMessage(content="You are an expert scientific problem solver. "
                                    "Verify if the given solution to the problem is correct. "
                                    "Provide a yes/no answer and explain your reasoning. "
                                    "If the solution is incorrect, explain why and provide the correct solution."),
                HumanMessage(content=f"Problem: {problem}\n\nProposed Solution: {solution}")
            ]
            
            # Generate verification
            response = self.llm.generate([messages])
            verification_text = response.generations[0][0].text.strip()
            
            # Check if the LLM thinks the solution is correct
            is_correct = ('yes' in verification_text[:50].lower() or 
                         'correct' in verification_text[:50].lower() or 
                         'solution is valid' in verification_text.lower())
            
            # Estimate confidence based on language used
            confidence = 0.5  # Default confidence
            if 'definitely' in verification_text.lower() or 'certainly' in verification_text.lower():
                confidence = 0.9
            elif 'probably' in verification_text.lower() or 'likely' in verification_text.lower():
                confidence = 0.7
            elif 'possibly' in verification_text.lower() or 'might' in verification_text.lower():
                confidence = 0.4
            elif 'unlikely' in verification_text.lower() or 'doubtful' in verification_text.lower():
                confidence = 0.2
            
            result['is_correct'] = is_correct
            result['confidence'] = confidence
            result['explanation'] = verification_text
            
        except Exception as e:
            logger.error(f"Error verifying solution: {str(e)}")
            result['error'] = str(e)
        
        return result
    
    def explain_solution(self, problem: str, solution: Any, domain: str, 
                        subdomain: Optional[str] = None, 
                        detail_level: str = 'medium') -> Dict[str, Any]:
        """
        Generate an explanation for a solution.
        
        Args:
            problem: The problem description
            solution: The solution to explain
            domain: The scientific domain
            subdomain: The specific subdomain (optional)
            detail_level: The level of detail (low, medium, high)
            
        Returns:
            Dict containing the explanation
        """
        result = {
            'explanation': None,
            'summary': None,
            'steps': [],
            'error': None
        }
        
        try:
            # Use LLM to explain the solution
            from langchain.schema import HumanMessage, SystemMessage
            
            # Adjust detail level
            detail_instructions = {
                'low': "Provide a brief overview of the solution approach.",
                'medium': "Explain the main steps and concepts in the solution.",
                'high': "Provide a detailed step-by-step explanation of the solution, including all relevant equations and calculations."
            }.get(detail_level.lower(), "Explain the main steps and concepts in the solution.")
            
            # Create prompt for explanation
            messages = [
                SystemMessage(content=f"You are an expert scientific educator. "
                                    f"{detail_instructions} "
                                    f"Make your explanation clear and educational, suitable for someone learning about this topic."),
                HumanMessage(content=f"Problem: {problem}\n\nSolution: {solution}")
            ]
            
            # Generate explanation
            response = self.llm.generate([messages])
            explanation_text = response.generations[0][0].text.strip()
            
            # Extract summary (first paragraph)
            summary = explanation_text.split('\n\n')[0]
            
            # Extract steps
            steps = []
            for line in explanation_text.split('\n'):
                if line.strip() and (line.strip().startswith('-') or 
                                   (line.strip()[0].isdigit() and line.strip()[1] in [')', '.', ':'])):
                    steps.append(line.strip())
            
            result['explanation'] = explanation_text
            result['summary'] = summary
            result['steps'] = steps
            
        except Exception as e:
            logger.error(f"Error explaining solution: {str(e)}")
            result['error'] = str(e)
        
        return result