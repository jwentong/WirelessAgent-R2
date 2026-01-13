# -*- coding: utf-8 -*-
"""
Enhanced WCHW Benchmark with Format-Aware Evaluation

This module provides fundamental improvements to the WCHW evaluation:

1. FORMAT-AWARE SCORING: Uses the QuestionAnalyzer to understand what 
   format the answer should be in, then scores accordingly.

2. INTELLIGENT EXTRACTION: Better extraction of answers from LLM responses,
   looking for "Final Answer:", boxed content, etc.

3. MULTI-STRATEGY MATCHING: Tries multiple matching strategies and takes
   the best score.

Key insight: ~30% of WCHW "errors" are actually format mismatches, not
calculation errors. By understanding the expected format, we can:
- Guide the LLM to output in the right format (via format hints)
- Score answers more fairly (format-aware matching)
- Reduce false negatives in evaluation
"""

import re
from typing import Callable, Optional, Tuple, Any

from benchmarks.wchw import WCHWBenchmark
from scripts.answer_format_classifier import (
    QuestionAnalyzer, 
    AnswerFormat, 
    AnswerFormatMatcher,
    FormatPrediction,
    create_format_enhanced_prompt
)
from scripts.logs import logger


class WCHWEnhancedBenchmark(WCHWBenchmark):
    """
    Enhanced WCHW Benchmark with format-aware evaluation.
    
    Improvements over base WCHWBenchmark:
    1. Pre-analyzes questions to predict expected answer format
    2. Uses AnswerFormatMatcher for smarter comparison
    3. Provides format hints to the workflow for better LLM guidance
    4. Handles more answer format variations
    """
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)
        self.question_analyzer = QuestionAnalyzer()
        self.format_matcher = AnswerFormatMatcher()
    
    def get_format_hint(self, question: str) -> str:
        """
        Get format hint for a question. Can be used by workflows to 
        guide the LLM's output format.
        """
        prediction = self.question_analyzer.analyze(question)
        return prediction.format_hint
    
    def extract_final_answer(self, response: str, format_type: AnswerFormat) -> str:
        """
        Extract the final answer from an LLM response.
        
        The LLM often produces long explanations before the final answer.
        This method tries multiple strategies to find the actual answer.
        """
        if response is None:
            return ""
        
        response = str(response)
        
        # Strategy 1: Look for explicit "Final Answer:" markers
        final_patterns = [
            r'(?:final\s+answer|answer|result)\s*[:=]\s*\**\s*([^\n]+)',
            r'✅\s*(?:final\s+answer|answer)?\s*[:=]?\s*\**\s*([^\n]+)',
            r'►\s*([^\n]+)',
            r'👉\s*([^\n]+)',
        ]
        
        for pattern in final_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Clean up markdown
                answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
                if len(answer) > 0 and len(answer) < 200:
                    return answer
        
        # Strategy 2: Look for LaTeX boxed content
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',
            r'\$\\boxed\{([^}]+)\}\$',
        ]
        
        for pattern in boxed_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1).strip()
        
        # Strategy 3: For numeric formats, find the last meaningful number
        if format_type in [AnswerFormat.PURE_NUMERIC, AnswerFormat.NUMERIC_WITH_UNIT,
                          AnswerFormat.SCIENTIFIC, AnswerFormat.RATIO, AnswerFormat.PERCENTAGE]:
            # Look for the last standalone number/expression
            lines = response.split('\n')
            for line in reversed(lines[-10:]):  # Check last 10 lines
                line = line.strip()
                if line.startswith('$') or '=' in line:
                    # This might be a formula/answer line
                    numbers = re.findall(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', line)
                    if numbers:
                        return line
        
        # Strategy 4: For short text formats, look for keywords
        if format_type == AnswerFormat.TEXT_SHORT:
            # Look for common short answer patterns
            patterns = [
                r'is\s+\**([A-Z]{2,}[A-Za-z]*)\**',  # Acronyms like FDMA
                r'called\s+(?:the\s+)?(\w+[\w\-]*)',
                r'is\s+(?:the\s+)?(\w+[\w\-]*(?:\s+\w+)?)\s*\.',
            ]
            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    return match.group(1).strip()
        
        # Strategy 5: For code sequences, find binary patterns
        if format_type == AnswerFormat.CODE_SEQUENCE:
            binary_match = re.search(r'[01]{8,}', response)
            if binary_match:
                return binary_match.group(0)
        
        # Fallback: return the full response (will be handled by matcher)
        return response
    
    def calculate_score_enhanced(
        self, 
        question: str,
        expected_output: Any, 
        prediction: Any,
        format_prediction: Optional[FormatPrediction] = None
    ) -> Tuple[float, str, Any]:
        """
        Enhanced scoring that considers multiple strategies.
        
        Returns: (score, explanation, extracted_prediction)
        """
        if format_prediction is None:
            format_prediction = self.question_analyzer.analyze(question)
        
        fmt = format_prediction.primary_format
        
        # Extract final answer from LLM response
        extracted = self.extract_final_answer(str(prediction), fmt)
        
        # Try multiple matching strategies
        strategies = []
        
        # Strategy 1: Use the new format-aware matcher
        score1, explanation1 = self.format_matcher.match_answer(
            question, str(expected_output), extracted, format_prediction
        )
        strategies.append((score1, explanation1, "format_matcher"))
        
        # Strategy 2: Try the original numeric scoring
        if fmt in [AnswerFormat.PURE_NUMERIC, AnswerFormat.NUMERIC_WITH_UNIT,
                  AnswerFormat.SCIENTIFIC, AnswerFormat.RATIO]:
            try:
                exp_val, _ = self.extract_number_with_unit(str(expected_output))
                pred_val, _ = self.extract_number_with_unit(extracted)
                
                if exp_val is not None and pred_val is not None:
                    score2 = self.calculate_numeric_score(exp_val, pred_val)
                    strategies.append((score2, f"numeric: {exp_val} vs {pred_val}", "numeric"))
            except Exception:
                pass
        
        # Strategy 3: Try formula matching for formula types
        if fmt == AnswerFormat.FORMULA:
            score3 = self.compare_formulas(str(expected_output), extracted)
            strategies.append((score3, "formula_match", "formula"))
        
        # Strategy 4: Try text matching as fallback
        score4 = self.compare_text_answers(str(expected_output), extracted)
        strategies.append((score4, "text_match", "text"))
        
        # Take the best score
        best_score, best_explanation, best_strategy = max(strategies, key=lambda x: x[0])
        
        logger.debug(f"Question format: {fmt.value}, Best strategy: {best_strategy}, Score: {best_score}")
        
        return best_score, best_explanation, extracted
    
    async def evaluate_problem(
        self, problem: dict, graph: Callable
    ) -> Tuple[str, str, Any, float, float]:
        """
        Evaluate a single problem with format-aware scoring.
        """
        input_text = problem["question"]
        expected_answer = problem["answer"]
        
        # Analyze question format
        format_prediction = self.question_analyzer.analyze(input_text)
        
        # Classify answer type (for backwards compatibility)
        answer_type = self.classify_answer_type(expected_answer)
        
        # For numeric/scientific types, normalize expected
        if answer_type in ['numeric', 'scientific']:
            expected_output, expected_method = self.normalize_answer(expected_answer, input_text)
            if expected_output is None:
                expected_output = expected_answer
                expected_method = 'raw'
        else:
            expected_output = expected_answer
            expected_method = answer_type

        try:
            output, cost = await self._generate_output(graph, input_text)
            
            # Use enhanced scoring
            score, explanation, extracted_output = self.calculate_score_enhanced(
                input_text, expected_output, output, format_prediction
            )
            
            if score < 1.0:
                extra_info = (f"type={format_prediction.primary_format.value}, "
                             f"method={explanation}, "
                             f"raw_answer={expected_answer}")
                self.log_mismatch(
                    input_text, 
                    expected_output, 
                    output, 
                    extracted_output,
                    extract_answer_code=extra_info
                )

            return input_text, output, expected_output, score, cost

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, 0.0


class FormatGuidedWorkflowEnhancer:
    """
    Utility class to enhance workflow prompts with format guidance.
    
    Usage:
        enhancer = FormatGuidedWorkflowEnhancer()
        enhanced_prompt = enhancer.enhance(question, original_prompt)
    """
    
    def __init__(self):
        self.analyzer = QuestionAnalyzer()
    
    def enhance(self, question: str, original_prompt: str = "") -> str:
        """Add format guidance to the prompt"""
        return create_format_enhanced_prompt(question, original_prompt)
    
    def get_format_suffix(self, question: str) -> str:
        """Get just the format guidance suffix to append to any prompt"""
        prediction = self.analyzer.analyze(question)
        
        return f"""

**OUTPUT FORMAT REQUIREMENT**:
{prediction.format_hint}
Examples: {', '.join(prediction.examples[:2]) if prediction.examples else 'N/A'}

Provide ONLY the final answer in the specified format. No explanations needed after the final answer.
"""


# Factory function to get enhanced benchmark
def get_enhanced_wchw_benchmark(name: str, file_path: str, log_path: str) -> WCHWEnhancedBenchmark:
    """Factory function to create enhanced WCHW benchmark"""
    return WCHWEnhancedBenchmark(name, file_path, log_path)


# Test the enhanced evaluation
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            "question": "What is the first-null bandwidth B of the multiplexed line signal?",
            "expected": "384.0",
            "predicted": "The first-null bandwidth B = R_b = 384 kHz for NRZ signaling.",
        },
        {
            "question": "The full name of an m-sequence is ____.",
            "expected": "maximal-length sequence",
            "predicted": "The full name of an m-sequence is **maximal-length sequence**.",
        },
        {
            "question": "Find the cross-correlation coefficient ρ between s1 and s2.",
            "expected": "-1",
            "predicted": "The coefficient ρ = -1 since the signals are antipodal.",
        },
        {
            "question": "Write the encoded sequence.",
            "expected": "11010010 00011101 01000010",
            "predicted": "The encoded sequence is: 11010010 00011101 01000010",
        },
    ]
    
    analyzer = QuestionAnalyzer()
    matcher = AnswerFormatMatcher()
    
    print("=" * 70)
    print("Enhanced WCHW Evaluation Test")
    print("=" * 70)
    
    for i, tc in enumerate(test_cases, 1):
        prediction = analyzer.analyze(tc["question"])
        score, explanation = matcher.match_answer(
            tc["question"], tc["expected"], tc["predicted"], prediction
        )
        
        print(f"\n[Test {i}]")
        print(f"Q: {tc['question'][:60]}...")
        print(f"Format: {prediction.primary_format.value}")
        print(f"Expected: {tc['expected'][:40]}")
        print(f"Predicted: {tc['predicted'][:40]}...")
        print(f"Score: {score:.2f} ({explanation})")
