# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : WCHW benchmark with multi-type answer evaluation
import re
import math
from typing import Callable, List, Optional, Tuple, Dict, Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from benchmarks.benchmark import BaseBenchmark
from scripts.logs import logger


class WCHWBenchmark(BaseBenchmark):
    """
    Enhanced benchmark for Wireless Communication Homework (WCHW) dataset.
    
    Supports multiple answer types:
    1. Numeric with units (e.g., "16 kbit/s", "44.8 kHz")
    2. Pure numeric (e.g., "1.54", "4800")
    3. Mathematical formulas/expressions (e.g., "1/(2τ_0)", "A^2 T")
    4. Scientific notation (e.g., "5.42e-6", "2.2×10^-8")
    5. Text/descriptive answers (e.g., phase sequences, waveform descriptions)
    6. LaTeX expressions (e.g., "$s_{FM}(t)=3\\cos...$")
    
    Scoring strategy:
    - Numeric answers: relative error based scoring
    - Formula answers: symbolic matching after normalization
    - Text answers: keyword/phrase matching
    """
    
    # Unit conversion factors to base units
    UNIT_MULTIPLIERS: Dict[str, float] = {
        # Frequency
        'ghz': 1e9, 'mhz': 1e6, 'khz': 1e3, 'hz': 1,
        # Data rate  
        'gbps': 1e9, 'gbit/s': 1e9, 'gbits/s': 1e9,
        'mbps': 1e6, 'mbit/s': 1e6, 'mbits/s': 1e6,
        'kbps': 1e3, 'kbit/s': 1e3, 'kbits/s': 1e3,
        'bps': 1, 'bit/s': 1, 'bits/s': 1,
        'baud': 1, 'kbaud': 1e3, 'mbaud': 1e6,
        # Power
        'kw': 1e3, 'w': 1, 'mw': 1e-3, 'uw': 1e-6, 'μw': 1e-6,
        'dbm': 1, 'dbw': 1, 'db': 1,  # dB values kept as-is
        # Time
        's': 1, 'ms': 1e-3, 'us': 1e-6, 'μs': 1e-6, 'ns': 1e-9,
        # Distance
        'km': 1e3, 'm': 1, 'cm': 1e-2, 'mm': 1e-3,
        # Spectral efficiency
        'bit/(s·hz)': 1, 'bit/s/hz': 1, 'bps/hz': 1,
        # Angle
        'deg': 1, 'rad': 1,
    }
    
    # Patterns for formula detection
    FORMULA_INDICATORS = [
        r'\\tau', r'\\omega', r'\\pi', r'\\alpha', r'\\beta', r'\\phi',
        r'\\cos', r'\\sin', r'\\log', r'\\exp', r'\\sqrt',
        r'_\{', r'\^', r'\\frac', r'\\le', r'\\ge',
        r'\$', r'tau_0', r'T_s', r'f_', r'R_',
    ]
    
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def classify_answer_type(self, answer: str) -> str:
        """
        Classify the answer type to determine scoring strategy.
        
        Returns:
            'numeric': Pure number or number with unit
            'scientific': Scientific notation
            'formula': Mathematical expression/formula
            'text': Text or descriptive answer
        """
        if answer is None:
            return 'unknown'
            
        answer_str = str(answer).strip()
        
        # Check for formula indicators
        for pattern in self.FORMULA_INDICATORS:
            if re.search(pattern, answer_str):
                # But also check if there's a dominant numeric component
                numbers = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', answer_str)
                if len(numbers) == 1 and len(answer_str) < 20:
                    # Likely just a number with some LaTeX formatting
                    return 'numeric'
                return 'formula'
        
        # Check for scientific notation
        if re.search(r'[-+]?\d+\.?\d*[eE][-+]?\d+', answer_str):
            return 'scientific'
        
        if re.search(r'[-+]?\d+\.?\d*\s*[×x\*]\s*10', answer_str):
            return 'scientific'
        
        # Check if it's primarily numeric
        numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', answer_str)
        
        if numbers:
            # Calculate how much of the string is numeric vs text
            total_num_chars = sum(len(n) for n in numbers)
            # Remove common unit text
            cleaned = answer_str.lower()
            for unit in self.UNIT_MULTIPLIERS.keys():
                cleaned = cleaned.replace(unit, '')
            cleaned = re.sub(r'[,\s\.\-\+]', '', cleaned)
            
            if len(cleaned) < 5 or total_num_chars > len(cleaned) * 0.3:
                return 'numeric'
        
        # Check if it's a long text answer
        if len(answer_str) > 100 or answer_str.count(' ') > 20:
            return 'text'
        
        # Short formula-like expressions without LaTeX
        if re.search(r'[a-zA-Z]+\s*=', answer_str) or '(' in answer_str:
            return 'formula'
        
        # Default: if there are any numbers, treat as numeric
        if numbers:
            return 'numeric'
        
        return 'text'

    def extract_unit(self, text: str) -> Tuple[Optional[str], float]:
        """Extract unit from text and return (unit_name, multiplier)."""
        text_lower = text.lower()
        
        # Check for units in order of specificity (longer patterns first)
        unit_patterns = sorted(self.UNIT_MULTIPLIERS.keys(), key=len, reverse=True)
        
        for unit in unit_patterns:
            if re.search(rf'\b{re.escape(unit)}\b', text_lower):
                return unit, self.UNIT_MULTIPLIERS[unit]
        
        return None, 1.0

    def extract_number_with_unit(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Extract the last number from text, considering units and scientific notation.
        Returns (value_in_base_unit, detected_unit)
        """
        if text is None:
            return None, None
            
        text_str = str(text)
        
        # Unicode superscript to digit mapping
        superscript_map = {'⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
                          '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
                          '⁻': '-', '⁺': '+'}
        
        # Normalize unicode superscripts
        normalized_text = text_str
        for sup, normal in superscript_map.items():
            normalized_text = normalized_text.replace(sup, normal)
        
        # Patterns for scientific notation
        sci_patterns = [
            # Multiplication notation: 2.2×10^-8 or 2.2 x 10^-8
            r'([-+]?\d+\.?\d*)\s*[×x\*]\s*10\s*\^?\s*[{\(]?\s*([-+]?\d+)\s*[}\)]?',
            # Standard scientific: 2.2e-8
            r'([-+]?\d+\.?\d*)[eE]([-+]?\d+)',
        ]
        
        for pattern in sci_patterns:
            matches = re.findall(pattern, normalized_text)
            if matches:
                mantissa_str, exp_str = matches[-1]
                try:
                    mantissa = float(mantissa_str)
                    exponent = int(exp_str)
                    value = mantissa * (10 ** exponent)
                    
                    # Look for unit after the scientific notation
                    full_match = re.search(pattern, normalized_text)
                    if full_match:
                        text_after = normalized_text[full_match.end():].strip()[:20]
                        unit, multiplier = self.extract_unit(text_after)
                        if unit and unit not in ['db', 'dbm', 'dbw']:  # Don't scale dB values
                            value = value * multiplier
                        return value, unit
                    return value, None
                except (ValueError, OverflowError):
                    continue
        
        # Fallback: Pattern for regular numbers
        number_pattern = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?'
        matches = re.findall(number_pattern, text_str)
        
        if not matches:
            return None, None
        
        # Get the last number
        last_number_str = matches[-1].replace(",", "")
        
        try:
            value = float(last_number_str)
        except ValueError:
            return None, None
        
        # Try to find unit near the number
        last_match_pos = text_str.rfind(last_number_str)
        text_after_number = text_str[last_match_pos + len(last_number_str):].strip()[:20]
        
        unit, multiplier = self.extract_unit(text_after_number)
        
        # Don't scale dB values
        if unit and unit not in ['db', 'dbm', 'dbw']:
            value = value * multiplier
            
        return value, unit

    def extract_number(self, text: str) -> Optional[float]:
        """Extract the last number from text (backward compatible)."""
        value, _ = self.extract_number_with_unit(text)
        return value

    def normalize_formula(self, formula: str) -> str:
        """
        Normalize a mathematical formula for comparison.
        Remove whitespace, convert to lowercase, standardize notation.
        """
        if not formula:
            return ""
        
        result = str(formula)
        
        # Remove LaTeX delimiters
        result = re.sub(r'\$+', '', result)
        
        # Standardize common LaTeX commands
        replacements = [
            (r'\\tau_0', 'tau0'),
            (r'\\tau', 'tau'),
            (r'\\omega', 'w'),
            (r'\\pi', 'pi'),
            (r'\\alpha', 'alpha'),
            (r'\\beta', 'beta'),
            (r'\\phi', 'phi'),
            (r'\\cos', 'cos'),
            (r'\\sin', 'sin'),
            (r'\\log', 'log'),
            (r'\\exp', 'exp'),
            (r'\\sqrt', 'sqrt'),
            (r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)'),  # \frac{a}{b} -> (a)/(b)
            (r'\\cdot', '*'),
            (r'\\times', '*'),
            (r'\\,', ''),
            (r'\\;', ''),
            (r'\\text\{[^}]*\}', ''),  # Remove \text{...}
            (r'\\mathrm\{[^}]*\}', ''),
        ]
        
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        # Remove spaces and convert to lowercase
        result = re.sub(r'\s+', '', result).lower()
        
        # Standardize brackets
        result = result.replace('[', '(').replace(']', ')')
        result = result.replace('{', '(').replace('}', ')')
        
        return result

    def compare_formulas(self, expected: str, predicted: str) -> float:
        """
        Compare two formula strings after normalization.
        Returns score between 0 and 1.
        """
        norm_expected = self.normalize_formula(expected)
        norm_predicted = self.normalize_formula(predicted)
        
        if not norm_expected or not norm_predicted:
            return 0.0
        
        # Exact match after normalization
        if norm_expected == norm_predicted:
            return 1.0
        
        # Check if one is contained in the other (partial match)
        if norm_expected in norm_predicted or norm_predicted in norm_expected:
            return 0.8
        
        # Character-level similarity (simple Jaccard-like)
        set_expected = set(norm_expected)
        set_predicted = set(norm_predicted)
        intersection = len(set_expected & set_predicted)
        union = len(set_expected | set_predicted)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Also check if key components match
        expected_vars = set(re.findall(r'[a-z]+[0-9]*', norm_expected))
        predicted_vars = set(re.findall(r'[a-z]+[0-9]*', norm_predicted))
        
        var_match = len(expected_vars & predicted_vars) / max(len(expected_vars), 1)
        
        # Combine scores
        final_score = 0.5 * similarity + 0.5 * var_match
        
        if final_score > 0.7:
            return 0.8
        elif final_score > 0.5:
            return 0.5
        
        return 0.0

    def compare_text_answers(self, expected: str, predicted: str) -> float:
        """
        Compare text-based answers using keyword matching.
        Returns score between 0 and 1.
        """
        if not expected or not predicted:
            return 0.0
        
        expected_lower = expected.lower()
        predicted_lower = predicted.lower()
        
        # Exact match
        if expected_lower.strip() == predicted_lower.strip():
            return 1.0
        
        # Extract key terms (numbers, technical terms)
        def extract_key_terms(text):
            # Numbers
            numbers = set(re.findall(r'[-+]?\d+\.?\d*', text))
            # Technical terms (words with letters)
            words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
            # Filter out common words
            stopwords = {'the', 'is', 'are', 'and', 'or', 'for', 'to', 'of', 'in', 'at', 'with', 'that', 'this'}
            words = words - stopwords
            return numbers, words
        
        exp_nums, exp_words = extract_key_terms(expected)
        pred_nums, pred_words = extract_key_terms(predicted)
        
        # Score based on matching components
        num_match = len(exp_nums & pred_nums) / max(len(exp_nums), 1) if exp_nums else 0.5
        word_match = len(exp_words & pred_words) / max(len(exp_words), 1) if exp_words else 0.5
        
        # For text answers, numbers are usually more important
        score = 0.6 * num_match + 0.4 * word_match
        
        if score > 0.8:
            return 1.0
        elif score > 0.6:
            return 0.8
        elif score > 0.4:
            return 0.5
        
        return 0.0

    def calculate_numeric_score(self, expected: float, predicted: float) -> float:
        """
        Calculate score for numeric answers with relative tolerance.
        """
        if predicted is None or expected is None:
            return 0.0
            
        # Handle zero expected output
        if abs(expected) < 1e-15:
            if abs(predicted) < 1e-15:
                return 1.0
            return 0.0
        
        # Calculate relative error
        relative_error = abs(expected - predicted) / abs(expected)
        
        # Exact or very close match (< 1% error)
        if relative_error < 0.01:
            return 1.0
        
        # Close match (< 5% error) - often due to rounding
        if relative_error < 0.05:
            return 0.9
        
        # Acceptable match (< 10% error)
        if relative_error < 0.10:
            return 0.7
        
        # Check for common unit conversion errors
        ratio = predicted / expected if expected != 0 else float('inf')
        
        # Off by factor of 1000 (kHz vs Hz, kbit/s vs bit/s)
        if 0.0009 < ratio < 0.0011 or 900 < ratio < 1100:
            logger.debug(f"Unit error (1000x): expected {expected}, got {predicted}")
            return 0.5
        
        # Off by factor of 1e6 (MHz vs Hz, Mbit/s vs bit/s)
        if 0.9e-6 < ratio < 1.1e-6 or 0.9e6 < ratio < 1.1e6:
            logger.debug(f"Unit error (1e6x): expected {expected}, got {predicted}")
            return 0.5
        
        # Off by factor of 2 (common formula error like B = Rs(1+α) vs Rs(1+α)/2)
        if 0.45 < ratio < 0.55 or 1.9 < ratio < 2.1:
            logger.debug(f"Factor of 2 error: expected {expected}, got {predicted}")
            return 0.3
            
        return 0.0

    def calculate_score(self, expected_output: Any, prediction: Any, 
                       answer_type: str = 'numeric') -> Tuple[float, Any]:
        """
        Calculate score based on answer type.
        
        Returns: (score, extracted_prediction)
        """
        if answer_type == 'numeric' or answer_type == 'scientific':
            # Convert to float if needed
            if isinstance(expected_output, str):
                expected_val, _ = self.extract_number_with_unit(expected_output)
            else:
                expected_val = expected_output
            
            if isinstance(prediction, str):
                predicted_val, _ = self.extract_number_with_unit(prediction)
            else:
                predicted_val = prediction
            
            score = self.calculate_numeric_score(expected_val, predicted_val)
            return score, predicted_val
        
        elif answer_type == 'formula':
            # Try numeric extraction first (some formulas have numeric answers)
            exp_val, _ = self.extract_number_with_unit(str(expected_output))
            pred_val, _ = self.extract_number_with_unit(str(prediction))
            
            if exp_val is not None and pred_val is not None:
                numeric_score = self.calculate_numeric_score(exp_val, pred_val)
                if numeric_score >= 0.7:
                    return numeric_score, pred_val
            
            # Fall back to formula comparison
            score = self.compare_formulas(str(expected_output), str(prediction))
            return score, prediction
        
        elif answer_type == 'text':
            score = self.compare_text_answers(str(expected_output), str(prediction))
            return score, prediction
        
        else:
            # Unknown type, try numeric first then text
            exp_val, _ = self.extract_number_with_unit(str(expected_output))
            pred_val, _ = self.extract_number_with_unit(str(prediction))
            
            if exp_val is not None and pred_val is not None:
                return self.calculate_numeric_score(exp_val, pred_val), pred_val
            
            return self.compare_text_answers(str(expected_output), str(prediction)), prediction

    def normalize_answer(self, answer: str, question: str) -> Tuple[Optional[float], str]:
        """
        Normalize answer considering units mentioned in the question.
        For numeric answers only.
        """
        value, unit = self.extract_number_with_unit(answer)
        
        if value is None:
            return None, "extraction_failed"
        
        a_unit, _ = self.extract_unit(answer)
        
        if a_unit:
            return value, f"extracted_{a_unit}"
        
        # Check if question specifies expected unit
        expected_unit_match = re.search(r'\bin\s+(\S+)', question.lower())
        
        if expected_unit_match:
            expected_unit_text = expected_unit_match.group(1).rstrip('?.,')
            if expected_unit_text in self.UNIT_MULTIPLIERS:
                mult = self.UNIT_MULTIPLIERS[expected_unit_text]
                # Don't scale dB values
                if expected_unit_text not in ['db', 'dbm', 'dbw']:
                    return value * mult, f"assumed_{expected_unit_text}"
        
        return value, "raw_number"

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, Any, float, float]:
        """
        Evaluate a single problem with type-aware scoring.
        """
        input_text = problem["question"]
        expected_answer = problem["answer"]
        
        # Classify answer type
        answer_type = self.classify_answer_type(expected_answer)
        
        # For numeric/scientific types, normalize
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
            
            # Calculate score based on answer type
            if answer_type in ['numeric', 'scientific']:
                predicted_output, pred_method = self.normalize_answer(str(output), input_text)
                if predicted_output is None:
                    predicted_output = self.extract_number(output)
                    pred_method = 'fallback'
            else:
                predicted_output = output
                pred_method = answer_type
            
            score, extracted_output = self.calculate_score(
                expected_output, predicted_output, answer_type
            )

            if score < 1.0:
                extra_info = f"type={answer_type}, exp_method={expected_method}, pred_method={pred_method}, raw_answer={expected_answer}"
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

    def get_result_columns(self) -> List[str]:
        return ["question", "prediction", "expected_output", "score", "cost"]
