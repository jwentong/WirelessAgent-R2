# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : operator demo of WirelessAgent

import asyncio
import concurrent.futures
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.async_llm import AsyncLLM
from scripts.logs import logger
from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, TextFormatter, CodeFormatter
from scripts.base_operator import BaseOperator, OperatorResult, OperatorMetrics
from scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
) # All BaseModel

from scripts.prompts.prompt import (
    ANSWER_GENERATION_PROMPT,
    FORMAT_PROMPT,
    MD_ENSEMBLE_PROMPT,
    PYTHON_CODE_VERIFIER_PROMPT,
    REFLECTION_ON_PUBLIC_TEST_PROMPT,
    REVIEW_PROMPT,
    REVISE_PROMPT,
    SC_ENSEMBLE_PROMPT,
)
from scripts.utils.code import (
    extract_test_cases_from_jsonl,
    test_case_2_test_function,
)

class Operator:
    def __init__(self, llm: AsyncLLM, name: str):
        self.name = name
        self.llm = llm

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """
        Fill a node using LLM with structured output
        
        Raises:
            - BadRequestError: Content inspection failed (should not retry)
            - RateLimitError: API rate limit exceeded (will auto-retry via tenacity)
            - FormatError: LLM output format is invalid
            - Other exceptions: Propagate to caller
        """
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        # Use the formatter with AsyncLLM - let exceptions propagate
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            # Fallback to direct call if no formatter is needed
            response = await self.llm(prompt)
            
        # Convert to expected format based on the original implementation
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None


class Custom(BaseOperator):
    """
    Custom operator for flexible LLM-based text generation.
    
    This operator allows users to provide custom instructions and input,
    making it highly versatile for various text processing tasks.
    
    Example:
        custom = Custom(llm)
        result = await custom(
            input="Paris is the capital of France.",
            instruction="Translate to Spanish: "
        )
        # result["response"] = "París es la capital de Francia."
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict[str, Any]:
        """Define input schema for Custom operator"""
        return {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input text to process"
                },
                "instruction": {
                    "type": "string",
                    "description": "The instruction/prompt for the LLM"
                }
            },
            "required": ["input", "instruction"],
            "description": "Custom operator accepts any instruction and input for flexible text generation"
        }
    
    def _get_output_schema(self) -> Dict[str, Any]:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The generated response from LLM"
                }
            },
            "description": "Returns a dictionary with 'response' key containing the LLM output"
        }
    
    async def _execute(self, input: str, instruction: str, **kwargs) -> Dict[str, Any]:
        """
        Execute custom instruction with input.
        
        Args:
            input: The input text to process
            instruction: The instruction/prompt for the LLM
            **kwargs: Additional arguments (ignored for backward compatibility)
        
        Returns:
            Dictionary with 'response' key containing the LLM output
        """
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """
        Helper method for LLM calls with formatting (inherited from Operator pattern).
        
        This method is kept for backward compatibility with the existing codebase.
        """
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        # Use the formatter with AsyncLLM
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            # Fallback to direct call if no formatter is needed
            response = await self.llm(prompt)
            
        # Convert to expected format
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None
    
    def _extract_cost(self, output: Any) -> float:
        """Extract LLM cost from output or LLM instance"""
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        return 0.0


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response


class CustomCodeGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "CustomCodeGenerate"):
        super().__init__(llm, name)

    async def __call__(self, problem, entry_point, instruction):
        prompt = instruction + problem
        response = await self._fill_node(GenerateOp, prompt, mode="code_fill", function_name=entry_point)
        return response


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

        prompt = SC_ENSEMBLE_PROMPT.format(question=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


def run_code(code):
    try:
        # Create a new global namespace
        global_namespace = {}

        disallowed_imports = [
            "os",
            "sys",
            "subprocess",
            "multiprocessing",
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "ggplot",
            "pylab",
            "tkinter",
            "PyQt5",
            "wx",
            "pyglet",
        ]

        # Check for prohibited imports
        for lib in disallowed_imports:
            if f"import {lib}" in code or f"from {lib}" in code:
                logger.info("Detected prohibited import: %s", lib)
                return "Error", f"Prohibited import: {lib} and graphing functionalities"

        # Use exec to execute the code
        exec(code, global_namespace)
        # Assume the code defines a function named 'solve'
        if "solve" in global_namespace and callable(global_namespace["solve"]):
            result = global_namespace["solve"]()
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
        # Create a class-level process pool, instead of creating a new one for each execution
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        """Ensure the process pool is closed when the object is destroyed"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)

    async def exec_code(self, code, timeout=30):
        """
        Asynchronously execute code and return an error if timeout occurs.
        """
        loop = asyncio.get_running_loop()

        try:
            # Use the class-level process pool
            future = loop.run_in_executor(self.process_pool, run_code, code)
            # Wait for the task to complete or timeout
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Only cancel this specific future, not the entire process pool
            future.cancel()
            # Force garbage collection
            import gc
            gc.collect()
            return "Error", "Code execution timed out"
        except concurrent.futures.process.BrokenProcessPool:
            # If the process pool is broken, recreate it
            self.process_pool.shutdown(wait=False)
            self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=1)
            return "Error", "Process pool broken, try again"
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

            # Force garbage collection after each iteration
            import gc
            gc.collect()

        return {"code": code, "output": output}

class Test(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Test"):
        super().__init__(llm, name)

    def exec_code(self, solution, entry_point):
        test_cases = extract_test_cases_from_jsonl(entry_point)

        fail_cases = []
        for test_case in test_cases:
            test_code = test_case_2_test_function(solution, test_case, entry_point)
            try:
                exec(test_code, globals())
            except AssertionError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                with open("tester.txt", "a") as f:
                    f.write("test_error of " + entry_point + "\n")
                error_infomation = {
                    "test_fail_case": {
                        "test_case": test_case,
                        "error_type": "AssertionError",
                        "error_message": str(e),
                        "traceback": tb_str,
                    }
                }
                fail_cases.append(error_infomation)
            except Exception as e:
                with open("tester.txt", "a") as f:
                    f.write(entry_point + " " + str(e) + "\n")
                return {"exec_fail_case": str(e)}
        if fail_cases != []:
            return fail_cases
        else:
            return "no error"

    async def __call__(self, problem, solution, entry_point, test_loop: int = 3):
        """
        "Test": {
        "description": "Test the solution with test cases, if the solution is correct, return 'no error'; if incorrect, reflect on the solution and the error information",
        "interface": "test(problem: str, solution: str, entry_point: str) -> str"
        }
        """
        for _ in range(test_loop):
            result = self.exec_code(solution, entry_point)
            if result == "no error":
                return {"result": True, "solution": solution}
            elif "exec_fail_case" in result:
                result = result["exec_fail_case"]
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass=f"executed unsuccessfully, error: \n {result}",
                    test_fail="executed unsucessfully",
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]
            else:
                prompt = REFLECTION_ON_PUBLIC_TEST_PROMPT.format(
                    problem=problem,
                    solution=solution,
                    exec_pass="executed successfully",
                    test_fail=result,
                )
                response = await self._fill_node(ReflectionTestOp, prompt, mode="code_fill")
                solution = response["response"]

        result = self.exec_code(solution, entry_point)
        if result == "no error":
            return {"result": True, "solution": solution}
        else:
            return {"result": False, "solution": solution}


class Format(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Format"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = FORMAT_PROMPT.format(problem_description=problem, solution=solution)
        response = await self._fill_node(FormatOp, prompt, mode)
        return response


class Review(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Review"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, mode: str = None):
        prompt = REVIEW_PROMPT.format(problem=problem, solution=solution)
        response = await self._fill_node(ReviewOp, prompt, mode="xml_fill")
        return response


class Revise(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "Revise"):
        super().__init__(llm, name)

    async def __call__(self, problem, solution, feedback, mode: str = None):
        prompt = REVISE_PROMPT.format(problem=problem, solution=solution, feedback=feedback)
        response = await self._fill_node(ReviseOp, prompt, mode="xml_fill")
        return response


class MdEnsemble(Operator):
    """
    Paper: Can Generalist Foundation Models Outcompete Special-Purpose Tuning? Case Study in Medicine
    Link: https://arxiv.org/abs/2311.16452
    """

    def __init__(self, llm: AsyncLLM, name: str = "MdEnsemble", vote_count: int = 5):
        super().__init__(llm, name)
        self.vote_count = vote_count

    @staticmethod
    def shuffle_answers(solutions: List[str]) -> Tuple[List[str], Dict[str, str]]:
        shuffled_solutions = solutions.copy()
        random.shuffle(shuffled_solutions)
        answer_mapping = {chr(65 + i): solutions.index(solution) for i, solution in enumerate(shuffled_solutions)}
        return shuffled_solutions, answer_mapping

    async def __call__(self, solutions: List[str], problem: str, mode: str = None):
        logger.info(f"solution count: {len(solutions)}")
        all_responses = []

        for _ in range(self.vote_count):
            shuffled_solutions, answer_mapping = self.shuffle_answers(solutions)

            solution_text = ""
            for index, solution in enumerate(shuffled_solutions):
                solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

            prompt = MD_ENSEMBLE_PROMPT.format(solutions=solution_text, question=problem)
            response = await self._fill_node(MdEnsembleOp, prompt, mode="xml_fill")

            answer = response.get("solution_letter", "A")
            answer = answer.strip().upper()

            if answer in answer_mapping:
                original_index = answer_mapping[answer]
                all_responses.append(original_index)

        most_frequent_index = Counter(all_responses).most_common(1)[0][0]
        final_answer = solutions[most_frequent_index]
        return {"solution": final_answer}


# ==================== ReAct Agent for Tool Calling (Plan B) ====================

class ReActAgent(BaseOperator):
    """
    ReAct (Reasoning + Acting) Agent Operator
    
    Based on paper: ReAct: Synergizing Reasoning and Acting in Language Models
    https://arxiv.org/abs/2210.03629
    
    Workflow:
    1. Thought: LLM thinks about what to do next
    2. Action: Choose and execute a tool
    3. Observation: Observe tool execution result
    4. Repeat 1-3 until final answer is reached
    
    Architecture:
    - Fixed Protocol: XML format requirements (immutable)
    - Strategy Prompt: Tool selection and reasoning guidelines (optimizable)
    - Runtime Context: Problem, history, tools (dynamic)
    
    Now inherits from BaseOperator for:
    - Automatic metrics collection (execution time, token usage)
    - Schema introspection for optimizer
    - Unified interface with other operators
    """
    
    def __init__(
        self, 
        llm: AsyncLLM, 
        tool_registry, 
        name: str = "ReActAgent",
        strategy_prompt: Optional[str] = None,
        enable_metrics: bool = True
    ):
        super().__init__(llm, enable_metrics=enable_metrics)
        self.name = name
        self.tool_registry = tool_registry
        
        # Use provided strategy or default
        if strategy_prompt is None:
            self.strategy_prompt = self._get_default_strategy()
        else:
            self.strategy_prompt = strategy_prompt
            logger.info(f"ReActAgent: Using custom strategy prompt ({len(strategy_prompt)} chars)")
    
    def _get_default_strategy(self) -> str:
        """Default strategy prompt (backward compatible)"""
        return """You are a helpful assistant that can use tools to answer questions.

Instructions:
1. Think about what information you need to answer the question
2. Decide on your next action:
   - If you need to use a tool, specify which tool and what arguments
   - If you have enough information to answer, provide the final answer

**ANSWER FORMAT REQUIREMENTS**:
- Provide CONCISE, DIRECT answers (not full sentences unless necessary)
- Match the expected answer format (e.g., if asking for a name, give just the name)
- Avoid unnecessary explanations in the final answer
- Examples:
  * "Are both X and Y orchids?" → Answer: "no" or "yes" (not "X is not an orchid")
  * "What profession do X and Y have?" → Answer: "novelist" (not "They are both novelists")
  * "Which season...?" → Answer: "2010" (not "The 2010 season")

Think step by step and be concise."""
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """
        Fill a node using LLM with structured output
        
        Inherited from Operator class to maintain compatibility.
        
        Raises:
            - BadRequestError: Content inspection failed (should not retry)
            - RateLimitError: API rate limit exceeded (will auto-retry via tenacity)
            - FormatError: LLM output format is invalid
            - Other exceptions: Propagate to caller
        """
        # Create appropriate formatter based on mode
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        # Use the formatter with AsyncLLM - let exceptions propagate
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            # Fallback to direct call if no formatter is needed
            response = await self.llm(prompt)
            
        # Convert to expected format based on the original implementation
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            # Return None if no specific formatter is needed
            return None
    
    def _get_input_schema(self) -> Dict[str, Any]:
        """Define input schema for ReActAgent"""
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The problem or question to solve using tools"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum number of ReAct iterations",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Whether to print detailed execution process",
                    "default": False
                }
            },
            "required": ["problem"],
            "description": "ReActAgent uses Think-Act-Observe loop to solve problems with tools"
        }
    
    def _get_output_schema(self) -> Dict[str, Any]:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Final answer to the problem"
                },
                "steps": {
                    "type": "array",
                    "description": "Execution history with thought-action-observation records",
                    "items": {
                        "type": "object",
                        "properties": {
                            "iteration": {"type": "integer"},
                            "thought": {"type": "string"},
                            "action": {"type": "string"},
                            "tool_args": {"type": "object"},
                            "observation": {"type": "object"}
                        }
                    }
                },
                "total_cost": {
                    "type": "number",
                    "description": "Total LLM API cost"
                },
                "reached_max_iterations": {
                    "type": "boolean",
                    "description": "Whether max iterations was reached without finding final answer"
                }
            },
            "description": "Returns final answer with execution trace and metadata"
        }
    
    async def _execute(
        self, 
        problem: str, 
        max_iterations: int = 5,
        verbose: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute ReAct loop (core implementation)
        
        Args:
            problem: Problem to solve
            max_iterations: Maximum number of iterations (default: 5)
            verbose: Whether to print detailed process
            **kwargs: Additional arguments (ignored, for BaseOperator compatibility)
        
        Returns:
            {
                "answer": str,
                "steps": List[Dict],  # Execution history
                "total_cost": float,
                "reached_max_iterations": bool (optional)
            }
        """
        from scripts.operator_an import ToolDecisionOp
        
        history = []
        consecutive_failures = 0  # Track consecutive tool failures for early stopping
        max_consecutive_failures = 2  # Stop if 2 consecutive tool calls fail (balanced tolerance)
        
        # Tool-forcing tracking
        forced_tool_use = False  # Track if we forced tool use in iteration 0
        
        for iteration in range(max_iterations):
            # Early stopping: if too many consecutive failures, give up
            if consecutive_failures >= max_consecutive_failures:
                logger.warning(f"ReActAgent: Early stopping after {consecutive_failures} consecutive failures")
                break
            
            # 1. Build prompt for current state
            prompt = self._build_prompt(problem, history)
            
            # 2. LLM decision
            try:
                decision = await self._fill_node(ToolDecisionOp, prompt, mode="xml_fill")
                
                # Validate decision has required fields
                if not isinstance(decision, dict):
                    raise ValueError(f"Invalid decision format: expected dict, got {type(decision)}")
                
                if 'thought' not in decision or 'action_type' not in decision:
                    raise ValueError(f"Missing required fields in decision. Got: {decision.keys()}")
                    
            except Exception as e:
                error_type = type(e).__name__
                
                # BadRequestError (400) - content inspection failed
                # Let it propagate immediately, don't try any fallback
                if "BadRequestError" in error_type or "data_inspection_failed" in str(e):
                    logger.error(f"Content inspection failed: {e}")
                    raise  # Propagate to ToolAgent
                
                # RateLimitError (429) - rate limit exceeded
                # Let it propagate, tenacity will retry
                if "RateLimitError" in error_type or "429" in str(e):
                    logger.error(f"Rate limit exceeded: {e}")
                    raise  # Propagate to ToolAgent
                
                # For other errors (FormatError, ValueError, etc.)
                logger.error(f"Decision error ({error_type}): {e}")
                logger.debug(f"Decision content: {decision if 'decision' in locals() else 'N/A'}")
                
                consecutive_failures += 1
                
                # Don't try LLM fallback - it might hit the same error
                # Just continue to next iteration or exit
                if consecutive_failures >= max_consecutive_failures:
                    raise  # Give up, let ToolAgent handle it
                
                # Skip this iteration and try again
                continue
            
            # Log action type (always show, not verbose-gated)
            action_type = decision.get('action_type', '').strip()
            
            # DEBUG: Log decision for analysis
            logger.info(f"ReActAgent iteration {iteration}: action_type='{action_type}'")
            
            # 3. Execute based on decision
            if action_type == 'final_answer':
                # NOTE: Tool-Forcing was removed after testing showed it degraded performance
                # The LLM's direct reasoning often outperforms forced tool usage for many problem types
                # Found final answer
                final_ans = decision.get('final_answer', '').strip()
                
                if not final_ans:
                    # Try to extract answer from thought field as fallback
                    thought = decision.get('thought', '').strip()
                    if thought:
                        # Look for common answer patterns in thought
                        import re
                        # Try to find numeric answers or key phrases
                        patterns = [
                            r'(?:answer|result|final|=)\s*[:=]?\s*([0-9.eE+\-]+)',
                            r'([0-9.eE+\-]+)\s*(?:Hz|kHz|MHz|GHz|dB|W|mW|bps|kbps|Mbps)?$',
                        ]
                        for pattern in patterns:
                            match = re.search(pattern, thought, re.IGNORECASE)
                            if match:
                                final_ans = match.group(1).strip()
                                logger.warning(f"ReActAgent: Extracted answer from thought: {final_ans}")
                                break
                    
                    if not final_ans:
                        logger.error("ReActAgent: Final answer is empty and no fallback found")
                        consecutive_failures += 1
                        continue
                
                logger.info(f"ReActAgent: Found answer after {iteration + 1} iteration(s)")
                
                # Check if tools were actually used
                tools_used = any('use_tool' in step.get('action', '').lower() or 
                               step.get('observation', {}).get('success') 
                               for step in history)
                
                if not tools_used and iteration > 0:
                    logger.warning(f"ReActAgent: Answer found but NO TOOLS USED! This may be inaccurate.")
                elif tools_used:
                    logger.info(f"ReActAgent: Tools were used ({len(history)} steps)")
                
                return {
                    "answer": final_ans,
                    "steps": history,
                    "total_cost": self.llm.get_usage_summary()["total_cost"],
                    "used_tools": tools_used
                }
            
            elif action_type == 'use_tool':
                tool_name = decision.get('tool_name', '').strip()
                tool_args_raw = decision.get('tool_args', '').strip()
                
                # Validate and parse tool_args
                # XML parser returns string, so we need to parse JSON if it's a string
                if not tool_args_raw:
                    tool_args = {}
                elif isinstance(tool_args_raw, str):
                    # Try to parse JSON string with automatic fixing
                    import json
                    import re
                    
                    # Remove common LLM mistakes before parsing
                    cleaned_args = tool_args_raw.strip()
                    
                    # Fix common mistake: double braces {{ }} instead of single { }
                    # This happens when LLM sees {{...}} in format string examples
                    if cleaned_args.startswith('{{') and cleaned_args.endswith('}}'):
                        logger.warning("ReActAgent: Detected double braces, fixing automatically")
                        cleaned_args = cleaned_args[1:-1]  # Remove outer braces
                    
                    # Fix LaTeX escapes in JSON strings
                    # Replace invalid JSON escapes like \( \) \[ \] \{ \} with escaped versions
                    # Only do this inside quoted strings to avoid breaking valid JSON syntax
                    def fix_latex_escapes(match):
                        """Fix LaTeX escape sequences inside JSON strings"""
                        text = match.group(0)
                        # Replace single backslash with double backslash for LaTeX commands
                        # But preserve valid JSON escapes like \" \n \t \\
                        text = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)
                        return text
                    
                    # Apply fix to all quoted strings in JSON
                    cleaned_args = re.sub(r'"([^"]*)"', fix_latex_escapes, cleaned_args)
                    
                    # Fix Python expressions in JSON values (e.g., "value": 1/(2*1) -> "value": 0.5)
                    # This happens when LLM puts math expressions directly in JSON instead of computing them
                    def evaluate_math_expressions(json_str):
                        """Replace simple math expressions in JSON values with their computed results"""
                        # Pattern: find numeric expressions after : including those with parentheses
                        # Match: : <spaces> <expression> where expression ends at , or }
                        # Expression can contain: digits, ., +, -, *, /, (, ), spaces
                        pattern = r':\s*([0-9.+\-*/() ]+?)(?=\s*[,}])'
                        
                        def eval_expr(match):
                            expr = match.group(1).strip()
                            
                            # Skip if it's already a simple number
                            try:
                                float(expr)
                                return match.group(0)  # Already a number, keep as-is
                            except ValueError:
                                pass  # Not a simple number, try to evaluate
                            
                            # Only evaluate if it contains operators and valid characters
                            if any(op in expr for op in ['+', '-', '*', '/']) and all(c in '0123456789.+-*/() ' for c in expr):
                                try:
                                    result = eval(expr, {"__builtins__": {}}, {})
                                    logger.info(f"ReActAgent: Auto-evaluated expression '{expr}' -> {result}")
                                    return f': {result}'
                                except Exception as e:
                                    logger.warning(f"ReActAgent: Failed to evaluate expression '{expr}': {e}")
                                    return match.group(0)  # Keep original if eval fails
                            
                            return match.group(0)  # Keep original if not a math expression
                        
                        return re.sub(pattern, eval_expr, json_str)
                    
                    cleaned_args = evaluate_math_expressions(cleaned_args)
                    
                    # If it looks like it might be malformed, try to fix it
                    if cleaned_args and not cleaned_args.startswith('{'):
                        # Might be missing braces - try to add them
                        logger.warning(f"ReActAgent: tool_args doesn't start with '{{': {repr(cleaned_args[:50])}")
                        # Don't auto-fix this - let it fail properly
                    
                    try:
                        tool_args = json.loads(cleaned_args)
                        if not isinstance(tool_args, dict):
                            logger.error(f"ReActAgent: Parsed tool_args is not a dict: {type(tool_args).__name__}")
                            consecutive_failures += 1
                            continue
                    except json.JSONDecodeError as e:
                        logger.error(f"ReActAgent: Failed to parse tool_args JSON: {e}")
                        logger.error(f"ReActAgent: Raw tool_args string (first 200 chars): {repr(cleaned_args[:200])}")
                        
                        # Try to help debug by showing context around error position
                        error_pos = e.pos if hasattr(e, 'pos') else 0
                        start = max(0, error_pos - 20)
                        end = min(len(cleaned_args), error_pos + 20)
                        if error_pos > 0:
                            logger.error(f"ReActAgent: Context around error: ...{repr(cleaned_args[start:end])}...")
                        
                        consecutive_failures += 1
                        continue
                else:
                    # tool_args_raw is already a dict (shouldn't happen with XML, but handle it)
                    tool_args = tool_args_raw
                
                if not tool_name:
                    logger.error("ReActAgent: Tool name is missing in decision")
                    consecutive_failures += 1
                    continue
                
                # Log tool usage (concise)
                logger.info(f"ReActAgent step {iteration + 1}: {tool_name}({list(tool_args.keys())})")
                
                # Execute tool
                observation = await self.tool_registry.execute_tool(tool_name, **tool_args)
                
                # Track consecutive failures for early stopping
                if observation.get('success', False):
                    consecutive_failures = 0  # Reset on success

                else:
                    consecutive_failures += 1
                    logger.warning(f"ReActAgent: Tool {tool_name} failed - {observation.get('error', 'Unknown error')}")
                
                # Record to history
                history.append({
                    "iteration": iteration + 1,
                    "thought": decision.get('thought', 'N/A'),
                    "action": f"Used {tool_name}",
                    "tool_args": tool_args,
                    "observation": observation
                })
            else:
                # Unknown action_type
                logger.warning(f"ReActAgent: Unknown action_type '{action_type}'. Skipping iteration.")
                consecutive_failures += 1
        
        # Reached max iterations or early stopped
        if consecutive_failures >= max_consecutive_failures:
            logger.warning(f"ReActAgent: Stopped after {consecutive_failures} consecutive failures")
        else:
            logger.info(f"ReActAgent: Reached max iterations ({max_iterations}) without final answer")
        
        # Try to generate answer based on history
        # Use simpler prompt to reduce token count and API load
        if history:
            # If we have some history, try to use it
            fallback_prompt = f"""Question: {problem}

Based on these research steps:
{self._format_history(history[-3:])}  # Only use last 3 steps to reduce tokens

Provide a concise answer."""
        else:
            # No history, just answer directly
            fallback_prompt = f"Answer concisely: {problem}"
        
        logger.info("ReActAgent: Attempting to generate answer from history...")
        try:
            fallback_answer = await self.llm(fallback_prompt)
            
            return {
                "answer": fallback_answer,
                "steps": history,
                "total_cost": self.llm.get_usage_summary()["total_cost"],
                "reached_max_iterations": True
            }
        except Exception as e:
            # If fallback also fails (e.g., 429 after all retries)
            logger.error(f"Fallback LLM call failed: {e}")
            # Return best available answer from history
            if history and len(history) > 0:
                last_obs = history[-1].get('observation', {})
                if last_obs.get('success'):
                    # Extract result from observation (different keys for different tools)
                    result_value = last_obs.get('result') or last_obs.get('count') or last_obs.get('valid') or last_obs.get('corrected_answer') or 'No clear result'
                    return {
                        "answer": f"Based on research: {result_value}",
                        "steps": history,
                        "total_cost": self.llm.get_usage_summary()["total_cost"],
                        "reached_max_iterations": True,
                        "error": "fallback_failed"
                    }
            
            # Absolute fallback: return error
            return {
                "answer": "Unable to generate answer due to API limitations.",
                "steps": history,
                "total_cost": self.llm.get_usage_summary()["total_cost"],
                "reached_max_iterations": True,
                "error": str(e)
            }
    
    def _build_prompt(self, problem: str, history: List[Dict]) -> str:
        """Build prompt for current iteration
        
        Architecture:
        1. Strategy Layer (optimizable): Tool selection, reasoning guidelines
        2. Runtime Context (dynamic): Tools, problem, history
        3. Fixed Protocol (immutable): XML format requirements
        """
        from scripts.prompts.prompt import REACT_AGENT_FIXED_PROTOCOL
        
        # Layer 3: Runtime Context (dynamic)
        tools_desc = self.tool_registry.get_concise_description()
        history_text = self._format_history(history)
        
        runtime_context = f"""
Available Tools:
{tools_desc}

Question: {problem}

Previous Steps:
{history_text if history else "None (this is the first step)"}
"""
        
        # Layer 2: Strategy (optimizable via constructor)
        strategy_section = self.strategy_prompt
        
        # Layer 1: Fixed Protocol (immutable)
        protocol_section = REACT_AGENT_FIXED_PROTOCOL
        
        # Compose final prompt
        prompt = f"""{strategy_section}

{runtime_context}

{protocol_section}
"""
        return prompt
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format history records"""
        if not history:
            return "None"
        
        formatted = []
        for step in history:
            obs = step['observation']
            status = "✅" if obs.get('success', False) else "❌"
            
            # Different tools return results with different keys
            # Try 'result', 'count', 'valid', or just show the whole observation
            if obs.get('success', False):
                result = obs.get('result') or obs.get('count') or obs.get('valid') or obs.get('corrected_answer') or str(obs)
            else:
                result = obs.get('error', 'Unknown error')
            
            formatted.append(f"""
Step {step['iteration']}:
  Thought: {step['thought']}
  Action: {step['action']}
  Arguments: {step['tool_args']}
  Result {status}: {result}
""")
        
        return "\n".join(formatted)


class AnswerValidator(BaseOperator):
    """
    Validate and correct answer based on question type.
    
    This operator uses both rule-based validation (fast) and LLM-based validation
    (for complex cases) to ensure the answer matches the expected format and type.
    
    Example:
        validator = AnswerValidator(llm)
        result = await validator(
            question="Which season did the show air?",
            answer="Season 1",
            expected_type="TEMPORAL_INFO: ordinal"
        )
        # result["answer"] = "first" (corrected)
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "AnswerValidator"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict[str, Any]:
        """Define input schema for validation"""
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The original question"
                },
                "answer": {
                    "type": "string",
                    "description": "The answer to validate"
                },
                "expected_type": {
                    "type": "string",
                    "description": "Expected answer type (e.g., 'TYPE_DESCRIPTION', 'ENTITY_NAME')"
                }
            },
            "required": ["question", "answer", "expected_type"]
        }
    
    def _get_output_schema(self) -> Dict[str, Any]:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "valid": {
                    "type": "boolean",
                    "description": "Whether the answer is valid"
                },
                "answer": {
                    "type": "string",
                    "description": "Original or corrected answer"
                },
                "correction_applied": {
                    "type": "boolean",
                    "description": "Whether correction was applied"
                }
            }
        }
    
    async def _execute(self, question: str, answer: str, expected_type: str, **kwargs) -> Dict[str, Any]:
        """
        Core validation logic.
        
        Args:
            question: Original question
            answer: Provided answer
            expected_type: Expected answer type (e.g., "TYPE_DESCRIPTION: ...")
        
        Returns:
            Dictionary with validation results
        """
        # Extract just the type name from expected_type
        type_name = expected_type.split(":")[0].strip() if ":" in expected_type else expected_type
        
        # Quick validation rules (no LLM needed for obvious cases)
        if type_name == "TEMPORAL_INFO":
            # Convert "Season 1" to "first", "Season 2" to "second", etc.
            if "season" in question.lower() and answer.lower().startswith("season "):
                season_map = {
                    "season 1": "first",
                    "season 2": "second", 
                    "season 3": "third",
                    "season 4": "fourth",
                    "season 5": "fifth",
                    "season 6": "sixth",
                    "season 7": "seventh",
                    "season 8": "eighth",
                }
                answer_lower = answer.lower()
                for season_str, ordinal in season_map.items():
                    if season_str in answer_lower:
                        return {
                            "valid": True,
                            "answer": ordinal,
                            "correction_applied": True
                        }
        
        # For other types, use LLM validation
        validation_prompt = f"""Question: {question}
Expected Answer Type: {expected_type}
Provided Answer: {answer}

Validation Rules:
1. If expected type is TYPE_DESCRIPTION, answer must be a category/type, not a specific name
2. If expected type is ALTERNATIVE_NAME, answer must be different from the name in question
3. If expected type is ACRONYM_EXPANSION, answer must be full words, not abbreviation
4. If expected type is DIFFERENCE_POINT, answer must be the difference itself, not full descriptions of both items
5. If expected type is ENTITY_NAME, answer should be a proper name
6. If answer is too long (>10 words) for a simple question, extract the core part

Task: 
- If answer matches expected type: output "VALID: [answer]"
- If answer doesn't match: output "CORRECTED: [corrected_answer]"

Provide only one line starting with VALID: or CORRECTED:"""

        try:
            response = await self.llm(validation_prompt)  # Fixed: use __call__ instead of .call()
            response = response.strip()
            
            if response.startswith("VALID:"):
                return {
                    "valid": True,
                    "answer": answer,  # Keep original
                    "correction_applied": False
                }
            elif response.startswith("CORRECTED:"):
                corrected = response.replace("CORRECTED:", "").strip()
                return {
                    "valid": True,
                    "answer": corrected,
                    "correction_applied": True
                }
            else:
                # Unexpected format, keep original
                logger.warning(f"AnswerValidator got unexpected response: {response}")
                return {
                    "valid": True,
                    "answer": answer,
                    "correction_applied": False
                }
        
        except Exception as e:
            logger.error(f"AnswerValidator error: {e}")
            # On error, return original answer
            return {
                "valid": True,
                "answer": answer,
                "correction_applied": False
            }


