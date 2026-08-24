import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.HotpotQA.workflows.template.operator_an import *
from workspace.HotpotQA.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator, ReActAgent, AnswerValidator
from scripts.base_operator import BaseOperator
from scripts.tools import (
    ToolRegistry, 
    WikipediaTool, 
    CalculatorTool,
    WikipediaPageTool,
    TextSearchTool,
    ComparisonTool,
    YearExtractorTool,
    AcronymExpanderTool  # NEW: Expand acronyms like GmbH, NASA
)



class Custom(BaseOperator):
    """
    Custom operator for flexible LLM-based transformations
    
    Generates any output based on customized input and instruction.
    Useful for formatting, analysis, or any custom transformation task.
    
    This is the workflow-specific version that inherits from BaseOperator
    for automatic metrics collection and metadata support.
    """
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict:
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
            "required": ["input", "instruction"]
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The generated response from LLM"
                }
            }
        }
    
    async def _execute(self, input: str, instruction: str, **kwargs) -> Dict:
        """Execute custom instruction on input"""
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Helper method for LLM calls with formatting"""
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            response = await self.llm(prompt)
            
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs):
        """Create appropriate formatter based on mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None
    
    def _extract_cost(self, output) -> float:
        """Extract LLM cost"""
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        return 0.0
    
class AnswerGenerate(Operator):
    """
    Direct answer generation operator
    
    Generates step-by-step reasoning and final answer without using tools.
    Used as fallback when tool-based approach fails.
    """
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        ## Generate answer with step-by-step reasoning
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
        return response

class ScEnsemble(Operator):
    """
    Self-Consistency Ensemble operator
    
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    
    Selects the most consistent solution from multiple candidate answers.
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, name)

    async def __call__(self, solutions: List[str]):
        ## Build solution comparison prompt
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        ## Use LLM to select most consistent answer
        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        ## Extract selected answer
        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}


# ==================== Tool-based Agent for HotpotQA ====================

class ToolAgent(BaseOperator):
    """
    Tool-augmented Agent for HotpotQA
    
    Uses ReActAgent with 7 Wikipedia and Calculator tools to answer questions
    requiring factual knowledge and multi-hop reasoning.
    
    Now inherits from BaseOperator for automatic metrics collection and
    metadata introspection to support MCTS optimization.
    
    Tools available:
    1. wikipedia_search - Search Wikipedia for topics
    2. wikipedia_page - Get full page content
    3. text_search - Search within text
    4. compare_entities - Compare two entities
    5. extract_year - Extract dates and years
    6. acronym_expander - Expand acronyms (GmbH, NASA, etc.)
    7. calculator - Perform calculations
    
    Features:
    - 9-type question analysis (ENTITY_NAME, TYPE_DESCRIPTION, etc.)
    - ReActAgent for tool-based reasoning
    - Answer validation and format correction
    - Intelligent fallback handling
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "ToolAgent"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        
        # Register basic tools
        self.tool_registry.register(WikipediaTool())
        self.tool_registry.register(CalculatorTool())
        
        # Register advanced tools for HotpotQA multi-hop reasoning
        self.tool_registry.register(WikipediaPageTool())       # Detailed page content
        self.tool_registry.register(TextSearchTool())          # Search within text
        self.tool_registry.register(ComparisonTool())          # Compare entities
        self.tool_registry.register(YearExtractorTool())       # Extract dates/years
        self.tool_registry.register(AcronymExpanderTool())     # Expand acronyms (GmbH, NASA, etc.)
        
        # Initialize ReActAgent
        self.react_agent = ReActAgent(llm, self.tool_registry, name="ReActAgent")
        
        # Initialize AnswerValidator (NEW)
        self.answer_validator = AnswerValidator(llm, name="AnswerValidator")
    
    def _get_input_schema(self) -> Dict:
        """Define input schema for ToolAgent"""
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The question to answer (typically multi-hop reasoning required)"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of reasoning steps",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 20
                }
            },
            "required": ["problem"],
            "description": "ToolAgent accepts a question and optionally max_steps for reasoning depth"
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The generated answer to the question"
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
                },
                "answer_corrected": {
                    "type": "boolean",
                    "description": "Whether answer was corrected by validator"
                }
            },
            "description": "Returns answer with reasoning steps and metadata"
        }
    
    async def _execute(self, problem: str, max_steps: int = 8, **kwargs) -> Dict:
        """
        Use tools to answer the question with intelligent fallback
        
        Args:
            problem: Question to answer
            max_steps: Maximum number of reasoning steps (default: 8 for HotpotQA multi-hop questions)
        
        Returns:
            {
                "answer": str,
                "steps": List[Dict],  # Tool calling history
                "total_cost": float,
                "used_fallback": bool  # Whether fallback was used
            }
        """
        ## Step 1: Analyze question type to determine expected answer format
        # Identifies one of 9 types: ENTITY_NAME, TYPE_DESCRIPTION, ATTRIBUTE_VALUE, 
        # LOCATION, ALTERNATIVE_NAME, COMPARISON_RESULT, DIFFERENCE_POINT, 
        # ACRONYM_EXPANSION, TEMPORAL_INFO
        question_type_analysis = None
        try:
            custom_op = Custom(self.llm, name="QuestionTypeAnalyzer")
            type_result = await custom_op(input=problem, instruction=ENHANCED_QUESTION_TYPE_ANALYSIS_PROMPT)
            if isinstance(type_result, dict) and 'response' in type_result:
                question_type_analysis = type_result['response']
                logger.info(f"Question type: {question_type_analysis}")
        except Exception as e:
            error_str = str(e)
            # Check if content inspection failed
            if "data_inspection_failed" in error_str or "inappropriate content" in error_str.lower():
                logger.info("Question type analysis skipped due to content restrictions (will proceed without type hint)")
            else:
                logger.warning(f"Question type analysis failed: {e}")
            question_type_analysis = None
        
        ## Step 2: Use ReActAgent with 7 tools to research and answer the question
        # Tools: wikipedia_search, wikipedia_page, text_search, compare_entities, 
        # extract_year, acronym_expander, calculator
        try:
            result = await self.react_agent(
                problem=problem,
                max_iterations=max_steps,
                verbose=False  # Set to True for detailed debugging
            )
            
            # Check if ReActAgent encountered errors
            if "error" in result or "reached_max_iterations" in result:
                logger.warning("ToolAgent: ReActAgent had issues, result quality may be low")
            
            ## Step 3: Validate and correct answer based on expected question type
            # Ensures answer matches the expected format (e.g., "Season 1" → "first")
            if question_type_analysis and 'answer' in result and result['answer']:
                try:
                    validation_result = await self.answer_validator(
                        question=problem,
                        answer=result['answer'],
                        expected_type=question_type_analysis
                    )
                    
                    if validation_result['correction_applied']:
                        logger.info(f"Answer corrected: '{result['answer']}' -> '{validation_result['answer']}'")
                        result['answer'] = validation_result['answer']
                        result['answer_corrected'] = True
                    else:
                        result['answer_corrected'] = False
                
                except Exception as e:
                    logger.warning(f"Answer validation failed: {e}")
                    result['answer_corrected'] = False
            
            result["used_fallback"] = False
            return result
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"ToolAgent failed with ReActAgent ({error_type}): {error_msg}")
            
            ## Fallback handling for different error types
            
            # 1. BadRequestError (400) - Content inspection failed
            if "BadRequestError" in error_type or "data_inspection_failed" in error_msg:
                logger.error("Content inspection failed. Cannot use tools for this question.")
                # Don't try AnswerGenerate fallback - it will hit the same content inspection
                return {
                    "answer": "This question cannot be processed due to content restrictions.",
                    "steps": [],
                    "total_cost": self.llm.get_usage_summary()["total_cost"],
                    "used_fallback": True,
                    "fallback_reason": "content_inspection_failed"
                }
            
            # 2. RateLimitError (429) - Rate limit exceeded
            if "RateLimitError" in error_type or "429" in error_msg:
                logger.error("Rate limit exceeded. Waiting before fallback...")
                # Wait a bit before trying fallback
                import asyncio
                await asyncio.sleep(5)
            
            # 3. For other errors, try AnswerGenerate fallback
            logger.info("Attempting AnswerGenerate fallback...")
            try:
                answer_gen = AnswerGenerate(self.llm)
                fallback_result = await answer_gen(input=problem)
                
                # Extract answer safely
                answer = None
                if isinstance(fallback_result, dict):
                    answer = fallback_result.get("answer") or fallback_result.get("response")
                else:
                    answer = str(fallback_result)
                
                if not answer:
                    answer = "Unable to generate answer"
                
                return {
                    "answer": answer,
                    "steps": [],
                    "total_cost": self.llm.get_usage_summary()["total_cost"],
                    "used_fallback": True,
                    "fallback_reason": f"{error_type}: {error_msg}"
                }
            except Exception as fallback_error:
                fallback_error_type = type(fallback_error).__name__
                logger.error(f"Fallback AnswerGenerate also failed ({fallback_error_type}): {fallback_error}")
                
                # If fallback fails with BadRequest, return safe message
                if "BadRequestError" in fallback_error_type:
                    return {
                        "answer": "Unable to answer due to content restrictions.",
                        "steps": [],
                        "total_cost": self.llm.get_usage_summary()["total_cost"],
                        "used_fallback": True,
                        "fallback_reason": "Both ReActAgent and AnswerGenerate blocked by content inspection"
                    }
                
                # For other fallback failures
                return {
                    "answer": "Unable to answer due to system errors.",
                    "steps": [],
                    "total_cost": self.llm.get_usage_summary()["total_cost"],
                    "used_fallback": True,
                    "fallback_reason": "Both ReActAgent and AnswerGenerate failed"
                }
    
    def _extract_cost(self, output: Any) -> float:
        """Extract LLM cost from ToolAgent output"""
        if isinstance(output, dict) and 'total_cost' in output:
            return float(output['total_cost'])
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        return 0.0
    
    def _extract_metadata(self, output: Any) -> Dict:
        """Extract metadata from ToolAgent output"""
        metadata = {}
        if isinstance(output, dict):
            # Tool calling statistics
            if 'steps' in output:
                metadata['num_steps'] = len(output['steps'])
                metadata['steps'] = output['steps']
            
            # Fallback information
            if 'used_fallback' in output:
                metadata['used_fallback'] = output['used_fallback']
            if 'fallback_reason' in output:
                metadata['fallback_reason'] = output['fallback_reason']
            
            # Answer correction flag
            if 'answer_corrected' in output:
                metadata['answer_corrected'] = output['answer_corrected']
        
        return metadata


