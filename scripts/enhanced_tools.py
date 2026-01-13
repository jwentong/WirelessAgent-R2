# -*- coding: utf-8 -*-
"""
Enhanced Tools for ToolAgent with RAG and Calculator Integration

This module provides two new tools that can be used by ToolAgent/ReActAgent:

1. TelecomFormulaRetriever - RAG tool for retrieving formulas from knowledge base
2. TelecomCalculator - Precise calculator for complex telecom functions

These tools are designed to work alongside PythonExecutor, allowing the optimizer
to choose the best approach for each problem:
- RAG for formula lookup and domain knowledge
- Calculator for precise numerical computation (erfc, Bessel, Marcum Q)
- PythonExecutor for custom code execution

The optimizer should learn when to use each tool based on problem characteristics.
"""

import os
import sys
from typing import Dict, Any, Optional, List
import math

# Import from scripts.tools (the .py file, not the directory)
# This contains BaseTool, ToolSchema, ToolParameter
import scripts.tools as tools_module
from scripts.logs import logger

BaseTool = tools_module.BaseTool
ToolSchema = tools_module.ToolSchema
ToolParameter = tools_module.ToolParameter

# Add telecom_tools directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
tools_dir = os.path.join(current_dir, 'telecom_tools')
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)


class TelecomFormulaRetriever(BaseTool):
    """
    RAG tool for retrieving telecommunications formulas from knowledge base.
    
    Use when:
    - Need to find the correct formula for a calculation
    - Unsure about exact formula structure or variables
    - Need domain-specific knowledge (BER formulas, fading models, etc.)
    - Want to verify approach before calculation
    
    Returns:
    - Relevant formulas with LaTeX and text representations
    - Variable definitions
    - Usage notes
    - Calculator function name (if available)
    
    Example:
        result = telecom_formula_retriever(query="BPSK BER formula coherent detection")
        # Returns BER = 0.5 * erfc(sqrt(Eb/N0)) with all details
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="telecom_formula_retriever",
            description="Search telecom formula knowledge base. Use when: (1) unsure about formula, (2) need variable definitions, (3) verify calculation approach. Categories: BER, capacity, fading, modulation, special_functions.",
            category="knowledge",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search keywords. E.g., 'BPSK BER', 'Rayleigh outage', 'Marcum Q'",
                    required=True
                ),
                ToolParameter(
                    name="top_k",
                    type="integer",
                    description="Number of results (default: 3)",
                    required=False,
                    default=3
                )
            ],
            usage_example='telecom_formula_retriever(query="BPSK BER coherent")',
            source="rag_knowledge_base",
            version="1.0"
        )
        super().__init__(schema)
        self._rag_tool = None
    
    def _get_rag_tool(self):
        """Lazy initialization of RAG tool."""
        if self._rag_tool is None:
            try:
                from scripts.telecom_tools.rag_tool import TelecomRAGTool
                kb_path = os.path.join(current_dir, 'telecom_tools', 'telecom_knowledge_base.json')
                self._rag_tool = TelecomRAGTool(kb_path)
            except ImportError as e:
                logger.error(f"Failed to import RAG tool: {e}")
                self._rag_tool = None
        return self._rag_tool
    
    async def execute(self, query: str = None, top_k: int = 3, **kwargs) -> Dict[str, Any]:
        """Execute RAG retrieval."""
        # Handle missing query - try alternative keys
        if not query or (isinstance(query, str) and not query.strip()):
            query = kwargs.get('search') or kwargs.get('keyword') or kwargs.get('formula')
            if not query:
                return {
                    "success": False,
                    "result": None,
                    "error": "Missing 'query' parameter. Usage: telecom_formula_retriever(query='BPSK BER')"
                }
        
        rag = self._get_rag_tool()
        
        if rag is None:
            return {
                "success": False,
                "result": None,
                "error": "RAG tool not available"
            }
        
        try:
            results = rag.retrieve(query, top_k=top_k)
            
            if not results:
                return {
                    "success": True,
                    "result": "No matching formulas found. Try broader keywords.",
                    "num_results": 0,
                    "formulas": []
                }
            
            # Format results for LLM consumption
            formatted_parts = []
            formula_list = []
            
            for i, r in enumerate(results, 1):
                formatted_parts.append(f"""
Formula {i}: {r.name}
  Category: {r.category}
  Formula: {r.formula_text}
  Variables: {', '.join([f'{k}={v}' for k,v in r.variables.items()]) if r.variables else 'N/A'}
  Notes: {r.notes}
  Calculator: {r.calculator_function or 'N/A'}
""")
                formula_list.append({
                    'id': r.id,
                    'name': r.name,
                    'formula_text': r.formula_text,
                    'calculator_function': r.calculator_function
                })
            
            return {
                "success": True,
                "result": "\n".join(formatted_parts),
                "num_results": len(results),
                "formulas": formula_list,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"TelecomFormulaRetriever error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


class TelecomPrecisionCalculator(BaseTool):
    """
    Precise calculator for telecommunications special functions.
    
    Use when:
    - Need to compute erfc, erf, Q-function accurately
    - Need Bessel functions (J_n, I_n for FM, Rician fading)
    - Need Marcum Q-function (Rician outage probability)
    - Need precise BER calculations for various modulations
    - Need Shannon capacity calculations
    - Need fading channel calculations (Rayleigh, Rician outage)
    
    This tool uses scipy for high-precision computation of special functions
    that LLMs typically get wrong.
    
    Example:
        result = telecom_calculator(
            operation="ber_bpsk_coherent",
            params={"Eb_N0_dB": 10}
        )
        # Returns precise BER value
    """
    
    AVAILABLE_OPERATIONS = [
        "erfc", "erf", "Q_function", "Q_inverse",
        "bessel_J", "bessel_I", "bessel_Y",
        "marcum_Q",
        "ber_bpsk_coherent", "ber_bfsk_coherent", "ber_bfsk_noncoherent", "ber_dpsk",
        "shannon_capacity",
        "rayleigh_outage_probability", "rayleigh_level_crossing_rate", "rayleigh_average_fade_duration",
        "rician_outage_probability",
        "fm_bessel_coefficients", "fm_carson_bandwidth"
    ]
    
    def __init__(self):
        # Create concise operation list for description
        key_ops = "erfc, Q_function, bessel_J, marcum_Q, ber_bpsk_coherent, shannon_capacity, rician_outage_probability"
        
        schema = ToolSchema(
            name="telecom_calculator",
            description=f"Precise telecom special functions. Use for: erfc, Bessel, Marcum Q (LLM often wrong). Key ops: {key_ops}",
            category="compute",
            parameters=[
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation name: erfc|Q_function|bessel_J|marcum_Q|ber_bpsk_coherent|shannon_capacity|rician_outage_probability|...",
                    required=True
                ),
                ToolParameter(
                    name="params",
                    type="object",
                    description="Dict of params. E.g., {'x':4} for Q_function, {'Eb_N0_dB':10} for BER, {'K':3,'gamma_threshold':1,'gamma_avg':10} for Rician",
                    required=True
                )
            ],
            usage_example='telecom_calculator(operation="ber_bpsk_coherent", params={"Eb_N0_dB": 10})',
            source="scipy_special",
            version="1.0"
        )
        super().__init__(schema)
        self._calculator = None
    
    def _get_calculator(self):
        """Lazy initialization of calculator."""
        if self._calculator is None:
            try:
                from scripts.telecom_tools.telecom_calculator import TelecomCalculator
                self._calculator = TelecomCalculator()
            except ImportError as e:
                logger.error(f"Failed to import calculator: {e}")
                self._calculator = None
        return self._calculator
    
    async def execute(self, operation: str = None, params: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute calculation."""
        # Handle missing operation
        if not operation:
            operation = kwargs.get('op') or kwargs.get('function') or kwargs.get('method')
            if not operation:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Missing 'operation' parameter. Available: {self.AVAILABLE_OPERATIONS[:5]}..."
                }
        
        # Handle missing params
        if params is None:
            params = {}
        
        # Handle case where LLM passes expression instead of operation/params
        if 'expression' in kwargs:
            return {
                "success": False,
                "result": None,
                "error": "telecom_calculator uses 'operation' and 'params', not 'expression'. Use calculator(expression=...) for simple math."
            }
        
        calc = self._get_calculator()
        
        if calc is None:
            # Fallback to basic math functions
            return self._fallback_calculate(operation, params)
        
        # Map operation to calculator method
        operation_map = {
            "erfc": lambda: calc.erfc(params.get('x', 0)),
            "erf": lambda: calc.erf(params.get('x', 0)),
            "Q_function": lambda: calc.Q_function(params.get('x', 0)),
            "Q_inverse": lambda: calc.Q_inverse(params.get('p', 0.5)),
            "bessel_J": lambda: calc.bessel_J(params.get('n', 0), params.get('x', 0)),
            "bessel_I": lambda: calc.bessel_I(params.get('n', 0), params.get('x', 0)),
            "bessel_Y": lambda: calc.bessel_Y(params.get('n', 0), params.get('x', 0)),
            "marcum_Q": lambda: calc.marcum_Q(params.get('a', 0), params.get('b', 0), params.get('M', 1)),
            "ber_bpsk_coherent": lambda: calc.ber_bpsk_coherent(params.get('Eb_N0_dB', 0)),
            "ber_bfsk_coherent": lambda: calc.ber_bfsk_coherent(params.get('Eb_N0_dB', 0)),
            "ber_bfsk_noncoherent": lambda: calc.ber_bfsk_noncoherent(params.get('Eb_N0_dB', 0)),
            "ber_dpsk": lambda: calc.ber_dpsk(params.get('Eb_N0_dB', 0)),
            "shannon_capacity": lambda: calc.shannon_capacity(
                params.get('bandwidth_Hz', 1e6), 
                params.get('snr_dB', 0)
            ),
            "rayleigh_outage_probability": lambda: calc.rayleigh_outage_probability(
                params.get('gamma_threshold', 1),
                params.get('gamma_avg', 10)
            ),
            "rayleigh_level_crossing_rate": lambda: calc.rayleigh_level_crossing_rate(
                params.get('rho', 1),
                params.get('f_D', 100)
            ),
            "rayleigh_average_fade_duration": lambda: calc.rayleigh_average_fade_duration(
                params.get('rho', 1),
                params.get('f_D', 100)
            ),
            "rician_outage_probability": lambda: calc.rician_outage_probability(
                params.get('K', 1),
                params.get('gamma_threshold', 1),
                params.get('gamma_avg', 10)
            ),
            "fm_bessel_coefficients": lambda: calc.fm_bessel_coefficients(
                params.get('beta', 1),
                params.get('n_max', 10)
            ),
            "fm_carson_bandwidth": lambda: calc.fm_carson_bandwidth(
                params.get('delta_f', 75000),
                params.get('f_m', 15000)
            ),
        }
        
        if operation not in operation_map:
            return {
                "success": False,
                "result": None,
                "error": f"Unknown operation: {operation}. Available: {list(operation_map.keys())}"
            }
        
        try:
            result_dict = operation_map[operation]()
            
            # Extract the value and explanation
            value = result_dict.get('value')
            explanation = result_dict.get('explanation', '')
            formula = result_dict.get('formula', '')
            
            return {
                "success": True,
                "result": value,
                "formatted": f"{value:.10e}" if isinstance(value, float) and abs(value) < 0.001 else f"{value:.6g}" if isinstance(value, (int, float)) else str(value),
                "explanation": explanation,
                "formula": formula,
                "full_result": result_dict,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"TelecomCalculator error: {e}")
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
    
    def _fallback_calculate(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback using basic Python math when scipy not available."""
        try:
            if operation == "erfc":
                result = math.erfc(params.get('x', 0))
            elif operation == "erf":
                result = math.erf(params.get('x', 0))
            elif operation == "Q_function":
                x = params.get('x', 0)
                result = 0.5 * math.erfc(x / math.sqrt(2))
            elif operation == "ber_bpsk_coherent":
                Eb_N0_dB = params.get('Eb_N0_dB', 0)
                Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
                result = 0.5 * math.erfc(math.sqrt(Eb_N0_linear))
            elif operation == "shannon_capacity":
                B = params.get('bandwidth_Hz', 1e6)
                snr_dB = params.get('snr_dB', 0)
                snr_linear = 10 ** (snr_dB / 10)
                result = B * math.log2(1 + snr_linear)
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Fallback calculation not available for: {operation}"
                }
            
            return {
                "success": True,
                "result": result,
                "formatted": f"{result:.6g}" if isinstance(result, float) else str(result),
                "note": "Computed using fallback (scipy not available)",
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Fallback calculation error: {str(e)}"
            }


def register_enhanced_tools(registry):
    """
    Register enhanced tools to the existing tool registry.
    
    Args:
        registry: ToolRegistry instance to add tools to
    
    Usage:
        from scripts.enhanced_tools import register_enhanced_tools
        register_enhanced_tools(tool_registry)
    """
    # Register RAG tool
    rag_tool = TelecomFormulaRetriever()
    registry.register(rag_tool)
    
    # Register Calculator tool
    calc_tool = TelecomPrecisionCalculator()
    registry.register(calc_tool)
    
    logger.info("✅ Registered enhanced tools: telecom_formula_retriever, telecom_calculator")


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Enhanced Tools")
    print("=" * 60)
    
    # Test RAG tool
    print("\n--- Testing TelecomFormulaRetriever ---")
    rag = TelecomFormulaRetriever()
    result = rag.execute(query="BPSK BER formula", top_k=2)
    print(f"Success: {result['success']}")
    print(f"Num results: {result.get('num_results', 0)}")
    print(f"Result:\n{result.get('result', 'N/A')}")
    
    # Test Calculator tool
    print("\n--- Testing TelecomPrecisionCalculator ---")
    calc = TelecomPrecisionCalculator()
    
    tests = [
        ("Q_function", {"x": 4.0}),
        ("ber_bpsk_coherent", {"Eb_N0_dB": 10}),
        ("shannon_capacity", {"bandwidth_Hz": 1e6, "snr_dB": 20}),
    ]
    
    for op, params in tests:
        result = calc.execute(operation=op, params=params)
        print(f"\n{op}({params}):")
        print(f"  Success: {result['success']}")
        print(f"  Result: {result.get('formatted', 'N/A')}")
        if result.get('explanation'):
            print(f"  Explanation: {result['explanation']}")
