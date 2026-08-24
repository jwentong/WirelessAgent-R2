import concurrent
import sys
import traceback
from typing import List, Optional, Dict, Any

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.MATH.workflows.template.operator_an import *
from workspace.MATH.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import asyncio


from scripts.operators import Operator, ReActAgent
from scripts.base_operator import BaseOperator
from scripts.tools import (
    ToolRegistry, 
    CalculatorTool,
    # NEW: Geometry tools
    GeometryConstraintChecker,
    GeometricFormulaTool,
    # NEW: Combinatorial tools
    CombinatorialEnumerator,
    ExpressionEnumerator,
    # NEW: Symbolic math tool
    SymbolicSolverTool,
    # NEW: Validation tools
    AnswerTypeValidator,
    ConstraintVerifierTool
)


class Custom(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, name)

    async def __call__(self, input, instruction):
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response

def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os", "sys", "subprocess", "multiprocessing",
            "matplotlib", "seaborn", "plotly", "bokeh", "ggplot",
            "pylab", "tkinter", "PyQt5", "wx", "pyglet"
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if 'solve' in global_namespace and callable(global_namespace['solve']):
            result = global_namespace['solve']()
            return "Success", str(result)
        else:
            return "Error", "Function 'solve' not found"
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
        return "Error", f"Execution error: {str(e)}\n{''.join(tb_str)}"
    

class Programmer(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Programmer"):
        super().__init__(llm, name)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            try:
                # Submit run_code task to the process pool
                future = loop.run_in_executor(executor, run_code, code)
                # Wait for the task to complete or timeout
                result = await asyncio.wait_for(future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                # Timeout, attempt to shut down the process pool
                executor.shutdown(wait=False, cancel_futures=True)
                return "Error", "Code execution timed out"
            except Exception as e:
                return "Error", f"Unknown error: {str(e)}"

    async def code_generate(self, problem, analysis, feedback, mode):
        """
        Asynchronous method to generate code.
        """
        prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
            problem=problem,
            analysis=analysis,
            feedback=feedback
        )
        response = await self._fill_node(CodeGenerateOp, prompt, mode, function_name="solve")
        return response

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def __call__(self, problem: str, analysis: str = "None"):
        """
        Call method, generate code and execute, retry up to 3 times.
        """
        code = None
        output = None
        feedback = ""
        for i in range(3):
            code_response = await self.code_generate(problem, analysis, feedback, mode="code_fill")
            code = code_response.get("code")
            if not code:
                return {"code": code, "output": "No code generated"}
            status, output = await self.exec_code(code)
            if status == "Success":
                return {"code": code, "output": output}
            else:
                print(f"Execution error on attempt {i + 1}, error message: {output}")
                feedback = (
                    f"\nThe result of the error from the code you wrote in the previous round:\n"
                    f"Code: {code}\n\nStatus: {status}, {output}"
                )
        return {"code": code, "output": output}


class ScEnsemble(Operator):
    """
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str], problem: str):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


class ToolAgent(BaseOperator):
    """
    Tool-augmented Agent for MATH problems
    
    Uses ReActAgent with math computation tools to solve mathematical problems
    requiring computational verification and symbolic calculations.
    
    Now inherits from BaseOperator for automatic metrics collection and
    metadata introspection to support MCTS optimization.
    
    Tools available:
    1. python_code_solver - Generate and execute Python code for complex calculations
    2. calculator - Evaluate mathematical expressions using SymPy
    
    Features:
    - ReActAgent for tool-based reasoning
    - Programmer tool for computational verification
    - Calculator for symbolic math operations
    - Intelligent fallback handling
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "ToolAgent", react_strategy: str = None):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Register Programmer as a tool
        from scripts.tools import ProgrammerTool
        programmer_tool = ProgrammerTool(llm)
        self.tool_registry.register(programmer_tool)
        
        # Register Calculator tool
        calculator_tool = CalculatorTool()
        self.tool_registry.register(calculator_tool)
        
        # NEW: Register Geometry tools
        geometry_checker = GeometryConstraintChecker()
        self.tool_registry.register(geometry_checker)
        
        geometry_formula = GeometricFormulaTool()
        self.tool_registry.register(geometry_formula)
        
        # NEW: Register Combinatorial tools
        combinatorial_enum = CombinatorialEnumerator()
        self.tool_registry.register(combinatorial_enum)
        
        expression_enum = ExpressionEnumerator()
        self.tool_registry.register(expression_enum)
        
        # NEW: Register Symbolic math tool
        symbolic_solver = SymbolicSolverTool()
        self.tool_registry.register(symbolic_solver)
        
        # NEW: Register Validation tools
        answer_validator = AnswerTypeValidator()
        self.tool_registry.register(answer_validator)
        
        constraint_verifier = ConstraintVerifierTool()
        self.tool_registry.register(constraint_verifier)
        
        # Use external strategy if provided, otherwise use default from op_prompt.py
        if react_strategy is None:
            react_strategy = REACT_STRATEGY_PROMPT
            logger.info("ToolAgent: Using default REACT_STRATEGY_PROMPT")
        else:
            logger.info(f"ToolAgent: Using custom react_strategy ({len(react_strategy)} chars)")
        
        # Initialize ReActAgent with strategy prompt
        self.react_agent = ReActAgent(
            llm, 
            self.tool_registry, 
            name="MATH_ReAct",
            strategy_prompt=react_strategy  # Now optimizable via parameter!
        )
        
        logger.info(f"ToolAgent initialized with {len(self.tool_registry.tools)} tools")
    
    def _get_input_schema(self) -> Dict:
        """Define input schema for ToolAgent"""
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The mathematical problem to solve"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of reasoning steps",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["problem"],
            "description": "ToolAgent accepts a math problem and optionally max_steps for reasoning depth"
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the math problem"
                },
                "steps": {
                    "type": "array",
                    "description": "Tool calling history (thought-action-observation)",
                    "items": {"type": "object"}
                },
                "total_cost": {
                    "type": "number",
                    "description": "Total LLM cost incurred"
                },
                "used_fallback": {
                    "type": "boolean",
                    "description": "Whether fallback (non-tool) method was used"
                }
            },
            "description": "Returns answer with reasoning steps and metadata"
        }
    
    async def _execute(self, problem: str, max_steps: int = 5, **kwargs) -> Dict:
        """
        Use tools to solve the math problem with intelligent fallback
        
        Args:
            problem: Mathematical problem to solve
            max_steps: Maximum number of reasoning steps (default: 5)
        
        Returns:
            {
                "answer": str,
                "steps": List[Dict],  # Tool calling history
                "total_cost": float,
                "used_fallback": bool
            }
        """
        try:
            # Use ReActAgent for tool-based problem solving
            result = await self.react_agent(
                problem=problem,
                max_iterations=max_steps,
                verbose=False
            )
            
            answer = result.get("answer", "")
            steps = result.get("steps", [])
            
            return {
                "answer": answer,
                "steps": steps,
                "total_cost": self._extract_cost(result),
                "used_fallback": False
            }
            
        except Exception as e:
            logger.error(f"ToolAgent error: {e}, falling back to direct LLM solving")
            
            # Fallback: Use direct LLM solving with SOLVE_PROMPT
            try:
                prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
                    problem=problem,
                    analysis="None",
                    feedback=""
                )
                response = await self.llm(prompt)
                
                return {
                    "answer": response if isinstance(response, str) else str(response),
                    "steps": [],
                    "total_cost": self._extract_cost({}),
                    "used_fallback": True
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "answer": "Error in solving",
                    "steps": [],
                    "total_cost": 0.0,
                    "used_fallback": True
                }
    
    def _extract_cost(self, output) -> float:
        """Extract LLM cost"""
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        return 0.0