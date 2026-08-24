import ast
import random
import sys
import traceback
from collections import Counter
from typing import Dict, List, Tuple

from scripts.base_operator import BaseOperator

from tenacity import retry, stop_after_attempt, wait_fixed

from scripts.formatter import BaseFormatter, FormatError, XmlFormatter, CodeFormatter, TextFormatter
from workspace.DROP.workflows.template.operator_an import *
from workspace.DROP.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger
import re


from scripts.operators import Operator


class Custom(BaseOperator):
    """
    Custom operator for DROP dataset
    
    Flexible LLM-based transformations for reading comprehension tasks.
    Now inherits from BaseOperator for automatic metrics collection.
    
    Note: Uses Operator's _fill_node method via composition.
    """
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        # Create an Operator instance to access _fill_node
        self._operator = Operator(llm, name)
    
    def _get_input_schema(self) -> Dict:
        """Define input schema"""
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text (context + question)"},
                "instruction": {"type": "string", "description": "Processing instruction"}
            },
            "required": ["input", "instruction"]
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Generated response"}
            }
        }
    
    async def _execute(self, input, instruction, **kwargs):
        """Execute custom transformation"""
        prompt = instruction + input
        response = await self._operator._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
    def _extract_cost(self, output) -> float:
        """Extract cost from output (DROP typically has simple outputs)"""
        # For DROP, cost is usually tracked at LLM level
        return 0.0


class AnswerValidator(BaseOperator):
    """
    Answer validator specifically for DROP dataset
    
    DROP answers can be:
    - Numbers (with/without commas): "1,234" or "1234"
    - Dates: "January 1, 2020" or "2020-01-01"
    - Text spans: direct quotes from context
    - Lists: "A, B, and C" or "A|B|C"
    
    This validator:
    1. Standardizes number formats
    2. Normalizes date formats
    3. Removes unnecessary articles/punctuation
    4. Validates answer format
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "AnswerValidator"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        self._operator = Operator(llm, name)
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "Raw answer to validate"},
                "question": {"type": "string", "description": "Original question for context"},
                "context": {"type": "string", "description": "Context passage (optional)"}
            },
            "required": ["answer", "question"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "validated_answer": {"type": "string", "description": "Cleaned and validated answer"},
                "is_valid": {"type": "boolean", "description": "Whether answer is valid"},
                "corrections": {"type": "array", "description": "List of corrections made"}
            }
        }
    
    async def _execute(self, answer: str, question: str, context: str = None, **kwargs) -> Dict:
        """Validate and clean DROP answer"""
        
        # Step 1: Basic cleaning
        cleaned = self._basic_clean(answer)
        
        # Step 2: Detect answer type
        answer_type = self._detect_answer_type(cleaned, question)
        
        # Step 3: Type-specific validation
        validated, corrections = self._type_specific_validation(cleaned, answer_type, question)
        
        # Step 4: If validation found issues, try LLM correction
        if corrections and context:
            try:
                corrected = await self._llm_correct(validated, question, context, corrections)
                if corrected:
                    validated = corrected
                    corrections.append("LLM correction applied")
            except Exception:
                pass  # Fall back to rule-based result
        
        return {
            "validated_answer": validated,
            "is_valid": len(corrections) == 0,
            "corrections": corrections,
            "answer_type": answer_type
        }
    
    def _basic_clean(self, answer: str) -> str:
        """Basic cleaning: trim, normalize whitespace"""
        import re
        answer = answer.strip()
        answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
        return answer
    
    def _detect_answer_type(self, answer: str, question: str) -> str:
        """Detect answer type from answer and question"""
        import re
        
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Number questions
        if any(word in question_lower for word in ['how many', 'how much', 'number of', 'percentage', 'percent']):
            if re.search(r'\d', answer):
                return "number"
        
        # Date questions
        if any(word in question_lower for word in ['when', 'what year', 'what date', 'which year']):
            if re.search(r'\d{4}', answer):  # Has a 4-digit year
                return "date"
        
        # List questions (multiple answers)
        if '|' in answer or (answer.count(',') >= 2 and 'and' in answer_lower):
            return "list"
        
        # Default to text
        return "text"
    
    def _type_specific_validation(self, answer: str, answer_type: str, question: str) -> tuple:
        """Validate based on answer type"""
        corrections = []
        
        if answer_type == "number":
            answer, num_corrections = self._validate_number(answer)
            corrections.extend(num_corrections)
        
        elif answer_type == "date":
            answer, date_corrections = self._validate_date(answer)
            corrections.extend(date_corrections)
        
        elif answer_type == "list":
            answer, list_corrections = self._validate_list(answer)
            corrections.extend(list_corrections)
        
        else:  # text
            answer, text_corrections = self._validate_text(answer)
            corrections.extend(text_corrections)
        
        return answer, corrections
    
    def _validate_number(self, answer: str) -> tuple:
        """Validate and normalize number format"""
        import re
        corrections = []
        
        # Remove commas from numbers
        if ',' in answer and re.search(r'\d,\d', answer):
            answer = answer.replace(',', '')
            corrections.append("Removed number commas")
        
        # Extract just the number if there's extra text
        match = re.search(r'(\d+(?:\.\d+)?)', answer)
        if match and match.group(0) != answer:
            answer = match.group(0)
            corrections.append("Extracted number from text")
        
        # Handle percentages
        if '%' in answer or 'percent' in answer.lower():
            match = re.search(r'(\d+(?:\.\d+)?)', answer)
            if match:
                answer = match.group(0)
                corrections.append("Normalized percentage")
        
        return answer, corrections
    
    def _validate_date(self, answer: str) -> tuple:
        """Validate and normalize date format"""
        import re
        corrections = []
        
        # Extract 4-digit year if present
        year_match = re.search(r'\b(1[0-9]{3}|2[0-9]{3})\b', answer)
        if year_match:
            year = year_match.group(0)
            # If answer is just the year or year with extra words
            if len(answer.split()) <= 3:
                answer = year
                if answer != year_match.group(0):
                    corrections.append("Extracted year from date")
        
        return answer, corrections
    
    def _validate_list(self, answer: str) -> tuple:
        """Validate list format"""
        corrections = []
        
        # Standardize to pipe-separated format
        if '|' not in answer:
            # Convert comma-separated to pipe
            import re
            # Remove "and" before last item
            answer = re.sub(r',?\s+and\s+', '|', answer)
            answer = answer.replace(',', '|')
            answer = '|'.join([item.strip() for item in answer.split('|')])
            corrections.append("Standardized list format")
        
        return answer, corrections
    
    def _validate_text(self, answer: str) -> tuple:
        """Validate text answer"""
        import re
        corrections = []
        
        # Remove common articles at the start
        if answer.lower().startswith(('the ', 'a ', 'an ')):
            answer = re.sub(r'^(the|a|an)\s+', '', answer, flags=re.IGNORECASE)
            corrections.append("Removed article")
        
        # Remove trailing punctuation
        if answer and answer[-1] in '.!?;:,':
            answer = answer[:-1]
            corrections.append("Removed trailing punctuation")
        
        return answer, corrections
    
    async def _llm_correct(self, answer: str, question: str, context: str, issues: List[str]) -> str:
        """Use LLM to correct answer if needed"""
        # Only use LLM for significant issues
        if len(issues) > 2:
            prompt = f"""Given the question and context, verify and correct this answer.

Question: {question}
Context: {context[:500]}...

Current answer: {answer}
Issues found: {', '.join(issues)}

Provide the corrected answer (just the answer, no explanation):
"""
            try:
                response = await self._operator._fill_node(GenerateOp, prompt, mode="single_fill")
                corrected = response.get("response", "").strip()
                if corrected and len(corrected) < 100:  # Sanity check
                    return corrected
            except Exception:
                pass
        
        return None
    
    def _extract_cost(self, output) -> float:
        """Extract LLM cost if LLM correction was used"""
        if output and output.get("corrections") and "LLM correction applied" in output.get("corrections", []):
            return 0.001  # Approximate cost for validation
        return 0.0


class AnswerGenerate(Operator):
    def __init__(self, llm: AsyncLLM, name: str = "AnswerGenerate"):
        super().__init__(llm, name)

    async def __call__(self, input: str, mode: str = None) -> Tuple[str, str]:
        prompt = ANSWER_GENERATION_PROMPT.format(input=input)
        response = await self._fill_node(AnswerGenerateOp, prompt, mode="xml_fill")
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

    async def __call__(self, solutions: List[str]):
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}