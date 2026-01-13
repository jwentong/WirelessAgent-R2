# -*- coding: utf-8 -*-
"""
Answer Format Classifier for WCHW Dataset

This module provides a fundamental improvement by:
1. Analyzing questions to predict expected answer format
2. Providing format hints to the LLM before solving
3. Classifying answer types for proper evaluation

Key insight: Many WCHW errors come from format mismatch, not calculation errors.
The LLM doesn't know if the expected answer is:
- A pure number (e.g., 384)
- A formula (e.g., π or 2·R_b)
- A unit-bearing value (e.g., 44.8 kHz)
- A text description (e.g., "FDMA")

This classifier analyzes the question to predict the expected format.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AnswerFormat(Enum):
    """Expected answer format types"""
    PURE_NUMERIC = "pure_numeric"           # Just a number: 384, 0.125
    NUMERIC_WITH_UNIT = "numeric_with_unit" # Number + unit: 44.8 kHz, 125 μs
    SCIENTIFIC = "scientific"               # Scientific notation: 2.89e-6
    FORMULA = "formula"                     # Mathematical expression: π, 2(1+β)f_m
    PERCENTAGE = "percentage"               # Percentage: 90%, 0.3
    TEXT_SHORT = "text_short"               # Single word/phrase: FDMA, m-sequence
    TEXT_LONG = "text_long"                 # Descriptive answer
    CODE_SEQUENCE = "code_sequence"         # Binary/code: 110100100001110100010001
    RATIO = "ratio"                         # Ratio or coefficient: -1, 0.9934


@dataclass
class FormatPrediction:
    """Prediction result for answer format"""
    primary_format: AnswerFormat
    confidence: float
    expected_unit: Optional[str] = None
    format_hint: str = ""
    examples: List[str] = None


class QuestionAnalyzer:
    """Analyzes questions to predict expected answer format"""
    
    # Keywords that indicate specific answer formats
    FORMAT_INDICATORS = {
        AnswerFormat.PURE_NUMERIC: [
            r"how many\b",
            r"what is the number of",
            r"count the",
            r"output only the numeric value",
            r"give the numerical value",
        ],
        AnswerFormat.NUMERIC_WITH_UNIT: [
            r"(?:in|expressed in)\s+(?:kHz|MHz|GHz|Hz)\b",
            r"(?:in|expressed in)\s+(?:kbit/s|Mbit/s|bit/s|bps)\b",
            r"(?:in|expressed in)\s+(?:ms|μs|ns|seconds?)\b",
            r"(?:in|expressed in)\s+(?:km|m|cm)\b",
            r"(?:in|expressed in)\s+(?:dB|dBm|dBW)\b",
            r"(?:in|expressed in)\s+(?:mW|μW|W|kW)\b",
            r"\(in\s+\w+\)",  # (in kbit/s), (in seconds), etc.
        ],
        AnswerFormat.SCIENTIFIC: [
            r"×10\^",
            r"\\times 10\^",
            r"e-\d+",
            r"very small",
            r"probability.*10\^-",
            r"in seconds.*small",
        ],
        AnswerFormat.FORMULA: [
            r"write the (?:expression|formula|equation)",
            r"give the (?:expression|formula|equation)",
            r"find the (?:expression|formula|equation)",
            r"state the (?:expression|formula|equation)",
            r"find the (?:modulating signal|impulse response|transfer function)",
            r"find.*\bf\(t\)\b",  # find f(t)
            r"find.*\bs\(t\)\b",  # find s(t)
            r"find.*\bh\(t\)\b",  # find h(t)
            r"derive",
        ],
        AnswerFormat.TEXT_SHORT: [
            r"(?:full )?name (?:of|for).*(?:is\s+)?____",
            r"is called\s+____",
            r"stands for\s+____",
            r"which (?:scheme|method|technique)",
            r"belongs to\s+____",
            r"(?:scheme|technique|method)\s+(?:used|is)\s+____",
            r"the\s+\w+\s+is\s+____\s*[.?]?\s*$",  # "The X is ____."
            r"are\s+____\s*[.?]?\s*$",
            r"____\s*[.?]?\s*$",  # ends with blank
        ],
        AnswerFormat.CODE_SEQUENCE: [
            r"write the (?:encoded|codeword|sequence)",
            r"find the (?:encoded|codeword|sequence)",
            r"output (?:sequence|stream|bits)",
            r"CMI|HDB3|NRZ|AMI",
            r"binary (?:sequence|stream|code)",
            r"encoded sequence",
        ],
        AnswerFormat.RATIO: [
            r"cross-correlation coefficient",
            r"correlation (?:coefficient|factor)",
            r"probability (?:to|of) remain",
            r"transition probability",
            r"what is p\(",
            r"stationary probability",
            r"modulation (?:index|depth|efficiency)",
        ],
        AnswerFormat.PERCENTAGE: [
            r"what percentage",
            r"what (?:fraction|ratio)",
            r"probability that",
            r"outage probability",
            r"probability\s+p_",
        ],
    }
    
    # Unit patterns for extraction
    UNIT_PATTERNS = {
        "frequency": r"(?:in\s+)?(kHz|MHz|GHz|Hz)",
        "data_rate": r"(?:in\s+)?(kbit/s|Mbit/s|Gbit/s|bit/s|bps|baud|kbaud|Mbaud)",
        "time": r"(?:in\s+)?(ms|μs|ns|s|seconds|microseconds|nanoseconds)",
        "power": r"(?:in\s+)?(mW|μW|W|kW|dBm|dBW|dB)",
        "distance": r"(?:in\s+)?(km|m|cm|mm)",
        "efficiency": r"(?:in\s+)?(bit/\(s·Hz\)|bit/s/Hz|bps/Hz)",
    }
    
    # Contextual patterns that modify format expectations
    CONTEXT_MODIFIERS = {
        "formula_context": [
            r"Carson's rule",
            r"Shannon's",
            r"Nyquist",
            r"the formula",
            r"using the relation",
        ],
        "calculation_context": [
            r"compute",
            r"calculate",
            r"find the (?:value|numerical)",
            r"what is the (?:value|numerical)",
            r"determine",
        ],
    }
    
    def analyze(self, question: str) -> FormatPrediction:
        """
        Analyze a question to predict the expected answer format.
        
        Args:
            question: The question text
            
        Returns:
            FormatPrediction with predicted format and hints
        """
        question_lower = question.lower()
        
        # Score each format based on indicator matches
        format_scores: Dict[AnswerFormat, float] = {fmt: 0.0 for fmt in AnswerFormat}
        
        for fmt, patterns in self.FORMAT_INDICATORS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    format_scores[fmt] += 1.0
        
        # Detect expected unit
        expected_unit = None
        for unit_type, pattern in self.UNIT_PATTERNS.items():
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                expected_unit = match.group(1)
                format_scores[AnswerFormat.NUMERIC_WITH_UNIT] += 0.5
                break
        
        # Check for fill-in-the-blank pattern
        if re.search(r"____\.?\s*$", question) or re.search(r"is\s+____", question):
            # Blank at end often expects short answer
            if format_scores[AnswerFormat.TEXT_SHORT] == 0:
                format_scores[AnswerFormat.TEXT_SHORT] += 0.3
            if format_scores[AnswerFormat.PURE_NUMERIC] == 0:
                format_scores[AnswerFormat.PURE_NUMERIC] += 0.3
        
        # Get the highest scoring format
        best_format = max(format_scores, key=format_scores.get)
        confidence = format_scores[best_format]
        
        # If no strong signal, default to pure numeric (most common)
        if confidence < 0.3:
            best_format = AnswerFormat.PURE_NUMERIC
            confidence = 0.3
        else:
            # Normalize confidence to 0-1 range
            confidence = min(1.0, confidence / 2.0)
        
        # Generate format hint
        format_hint = self._generate_format_hint(best_format, expected_unit)
        examples = self._get_format_examples(best_format)
        
        return FormatPrediction(
            primary_format=best_format,
            confidence=confidence,
            expected_unit=expected_unit,
            format_hint=format_hint,
            examples=examples
        )
    
    def _generate_format_hint(self, fmt: AnswerFormat, unit: Optional[str]) -> str:
        """Generate a hint to guide the LLM's output format"""
        hints = {
            AnswerFormat.PURE_NUMERIC: 
                "Output a pure number only (no units). Example: 384 or 0.125",
            AnswerFormat.NUMERIC_WITH_UNIT: 
                f"Output a number with unit{(' in ' + unit) if unit else ''}. Example: 44.8 kHz",
            AnswerFormat.SCIENTIFIC:
                "Output in scientific notation. Example: 2.89e-6 or 5×10^8",
            AnswerFormat.FORMULA:
                "Output a mathematical formula/expression. Use LaTeX if needed. Example: π or 2(1+β)f_m",
            AnswerFormat.TEXT_SHORT:
                "Output a short answer (word or phrase). Example: FDMA or m-sequence",
            AnswerFormat.TEXT_LONG:
                "Provide a brief descriptive answer explaining the concept.",
            AnswerFormat.CODE_SEQUENCE:
                "Output the binary/code sequence directly. Example: 11010010",
            AnswerFormat.RATIO:
                "Output a ratio or coefficient (number between -1 and 1 typically). Example: -1 or 0.9934",
            AnswerFormat.PERCENTAGE:
                "Output a probability or percentage. Example: 0.3 or 30%",
        }
        return hints.get(fmt, "Provide a concise answer.")
    
    def _get_format_examples(self, fmt: AnswerFormat) -> List[str]:
        """Get example answers for each format"""
        examples = {
            AnswerFormat.PURE_NUMERIC: ["384", "0.125", "100", "1.54"],
            AnswerFormat.NUMERIC_WITH_UNIT: ["44.8 kHz", "125 μs", "50 MHz", "6.58 dB"],
            AnswerFormat.SCIENTIFIC: ["2.89e-6", "5×10^8", "4.88e-06"],
            AnswerFormat.FORMULA: ["π", "2(1+β)f_m", "s(t)=A cos(ωt)", "R_b/B"],
            AnswerFormat.TEXT_SHORT: ["FDMA", "m-sequence", "water-filling"],
            AnswerFormat.CODE_SEQUENCE: ["11010010", "110001..."],
            AnswerFormat.RATIO: ["-1", "0.9934", "0.0066"],
            AnswerFormat.PERCENTAGE: ["0.3", "0.2", "90%"],
        }
        return examples.get(fmt, [])


class AnswerFormatMatcher:
    """
    Enhanced answer matching that considers format expectations.
    
    This class provides smarter answer matching by:
    1. Using format prediction to guide comparison
    2. Handling format variations (LaTeX vs plain, unit variations)
    3. Providing partial credit for nearly-correct answers
    """
    
    def __init__(self):
        self.analyzer = QuestionAnalyzer()
    
    def match_answer(
        self, 
        question: str,
        expected: str,
        predicted: str,
        format_prediction: Optional[FormatPrediction] = None
    ) -> Tuple[float, str]:
        """
        Match predicted answer against expected answer.
        
        Returns:
            Tuple of (score, explanation)
        """
        if format_prediction is None:
            format_prediction = self.analyzer.analyze(question)
        
        fmt = format_prediction.primary_format
        
        # Normalize both answers
        expected_norm = self._normalize_answer(expected, fmt)
        predicted_norm = self._normalize_answer(predicted, fmt)
        
        # Format-specific matching
        if fmt in [AnswerFormat.PURE_NUMERIC, AnswerFormat.NUMERIC_WITH_UNIT, 
                   AnswerFormat.SCIENTIFIC, AnswerFormat.RATIO, AnswerFormat.PERCENTAGE]:
            return self._match_numeric(expected_norm, predicted_norm)
        elif fmt == AnswerFormat.FORMULA:
            return self._match_formula(expected_norm, predicted_norm)
        elif fmt == AnswerFormat.CODE_SEQUENCE:
            return self._match_sequence(expected_norm, predicted_norm)
        else:
            return self._match_text(expected_norm, predicted_norm)
    
    def _normalize_answer(self, answer: str, fmt: AnswerFormat) -> str:
        """Normalize answer based on expected format"""
        if answer is None:
            return ""
        
        answer = str(answer).strip()
        
        # Remove markdown formatting
        answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', answer)
        answer = re.sub(r'\$([^$]+)\$', r'\1', answer)
        
        # For numeric formats, try to extract the final answer
        if fmt in [AnswerFormat.PURE_NUMERIC, AnswerFormat.NUMERIC_WITH_UNIT, 
                   AnswerFormat.SCIENTIFIC, AnswerFormat.RATIO, AnswerFormat.PERCENTAGE]:
            # Look for "Final Answer:" or boxed content
            final_match = re.search(
                r'(?:final answer|answer|result)[\s:]+([^\n]+)',
                answer, re.IGNORECASE
            )
            if final_match:
                answer = final_match.group(1).strip()
            
            # Look for boxed LaTeX
            boxed_match = re.search(r'\\boxed\{([^}]+)\}', answer)
            if boxed_match:
                answer = boxed_match.group(1)
        
        return answer
    
    def _match_numeric(
        self, expected: str, predicted: str
    ) -> Tuple[float, str]:
        """Match numeric answers with tolerance"""
        try:
            exp_val = self._extract_number(expected)
            pred_val = self._extract_number(predicted)
            
            if exp_val is None or pred_val is None:
                return 0.0, "Could not extract numeric values"
            
            if exp_val == 0:
                if pred_val == 0:
                    return 1.0, "Exact match (zero)"
                return 0.0, f"Expected 0, got {pred_val}"
            
            # Relative error calculation
            rel_error = abs(exp_val - pred_val) / abs(exp_val)
            
            if rel_error < 0.01:  # <1%
                return 1.0, "Exact match"
            elif rel_error < 0.05:  # <5%
                return 0.9, f"Close match (error: {rel_error*100:.1f}%)"
            elif rel_error < 0.10:  # <10%
                return 0.7, f"Approximate match (error: {rel_error*100:.1f}%)"
            else:
                # Check for common factor errors
                ratio = pred_val / exp_val if exp_val != 0 else float('inf')
                if abs(ratio - 1000) < 100 or abs(ratio - 0.001) < 0.0001:
                    return 0.5, "Off by factor of 1000 (unit conversion error)"
                elif abs(ratio - 2) < 0.1 or abs(ratio - 0.5) < 0.05:
                    return 0.3, "Off by factor of 2"
                return 0.0, f"Value mismatch: expected {exp_val}, got {pred_val}"
                
        except Exception as e:
            return 0.0, f"Error in numeric matching: {str(e)}"
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        if text is None:
            return None
        
        text = str(text)
        
        # Handle scientific notation
        sci_match = re.search(
            r'([-+]?\d+\.?\d*)\s*[×x\*]?\s*10\s*\^?\s*[{(]?\s*([-+]?\d+)\s*[})]?',
            text
        )
        if sci_match:
            try:
                mantissa = float(sci_match.group(1))
                exp = int(sci_match.group(2))
                return mantissa * (10 ** exp)
            except:
                pass
        
        # Handle e notation
        e_match = re.search(r'([-+]?\d+\.?\d*)[eE]([-+]?\d+)', text)
        if e_match:
            try:
                mantissa = float(e_match.group(1))
                exp = int(e_match.group(2))
                return mantissa * (10 ** exp)
            except:
                pass
        
        # Handle regular numbers (get the last one)
        numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1].replace(',', ''))
            except:
                pass
        
        return None
    
    def _match_formula(
        self, expected: str, predicted: str
    ) -> Tuple[float, str]:
        """Match formula/expression answers"""
        # Normalize formulas
        exp_norm = self._normalize_formula(expected)
        pred_norm = self._normalize_formula(predicted)
        
        # Exact match
        if exp_norm == pred_norm:
            return 1.0, "Exact formula match"
        
        # Check if they're numerically equivalent (e.g., π vs 3.14159)
        exp_num = self._formula_to_number(exp_norm)
        pred_num = self._formula_to_number(pred_norm)
        
        if exp_num is not None and pred_num is not None:
            rel_error = abs(exp_num - pred_num) / abs(exp_num) if exp_num != 0 else float('inf')
            if rel_error < 0.01:
                return 0.9, "Numerically equivalent"
        
        # Check for structural similarity (ignore whitespace, some notation differences)
        if self._formulas_structurally_similar(exp_norm, pred_norm):
            return 0.8, "Structurally similar formula"
        
        return 0.0, f"Formula mismatch: expected {expected}"
    
    def _normalize_formula(self, formula: str) -> str:
        """Normalize mathematical formula for comparison"""
        if formula is None:
            return ""
        
        formula = str(formula)
        
        # Remove LaTeX formatting
        formula = formula.replace('\\', '')
        formula = formula.replace('{', '').replace('}', '')
        formula = formula.replace('$', '')
        formula = formula.replace('\\,', '')
        formula = formula.replace('\\;', '')
        
        # Normalize common variations
        formula = formula.replace('×', '*')
        formula = formula.replace('·', '*')
        formula = formula.replace('–', '-')
        formula = formula.replace('−', '-')
        
        # Remove whitespace
        formula = re.sub(r'\s+', '', formula)
        
        return formula.lower()
    
    def _formula_to_number(self, formula: str) -> Optional[float]:
        """Try to evaluate formula to a number"""
        import math
        
        # Handle special constants
        replacements = {
            'pi': str(math.pi),
            'π': str(math.pi),
            'e': str(math.e),
        }
        
        formula_eval = formula
        for symbol, value in replacements.items():
            formula_eval = formula_eval.replace(symbol, value)
        
        try:
            # Safe eval for simple expressions
            return float(eval(formula_eval, {"__builtins__": {}}, {"sqrt": math.sqrt}))
        except:
            return None
    
    def _formulas_structurally_similar(self, f1: str, f2: str) -> bool:
        """Check if two formulas have similar structure"""
        # Simple heuristic: similar length and similar characters
        if abs(len(f1) - len(f2)) > len(f1) * 0.3:
            return False
        
        # Check character overlap
        chars1 = set(f1)
        chars2 = set(f2)
        overlap = len(chars1 & chars2) / max(len(chars1 | chars2), 1)
        
        return overlap > 0.7
    
    def _match_sequence(
        self, expected: str, predicted: str
    ) -> Tuple[float, str]:
        """Match code/binary sequences"""
        # Remove spaces and formatting
        exp_clean = re.sub(r'[^01]', '', expected)
        pred_clean = re.sub(r'[^01]', '', predicted)
        
        if exp_clean == pred_clean:
            return 1.0, "Exact sequence match"
        
        # Check for partial match
        if len(exp_clean) > 0 and len(pred_clean) > 0:
            # Calculate Hamming distance for same-length strings
            if len(exp_clean) == len(pred_clean):
                hamming = sum(c1 != c2 for c1, c2 in zip(exp_clean, pred_clean))
                accuracy = 1 - (hamming / len(exp_clean))
                if accuracy > 0.9:
                    return 0.8, f"Sequence nearly correct ({accuracy*100:.0f}% match)"
        
        return 0.0, "Sequence mismatch"
    
    def _match_text(
        self, expected: str, predicted: str
    ) -> Tuple[float, str]:
        """Match text answers with improved heuristics"""
        # Normalize for comparison
        exp_norm = expected.lower().strip()
        pred_norm = predicted.lower().strip()
        
        # Remove markdown formatting from predicted
        pred_norm = re.sub(r'\*\*([^*]+)\*\*', r'\1', pred_norm)
        pred_norm = re.sub(r'`([^`]+)`', r'\1', pred_norm)
        
        # Exact match
        if exp_norm == pred_norm:
            return 1.0, "Exact text match"
        
        # Check if expected is contained in predicted (common case)
        if exp_norm in pred_norm:
            return 0.95, "Answer contained in response"
        
        # Check if expected is a hyphenated version
        exp_hyphen = exp_norm.replace(' ', '-')
        exp_underscore = exp_norm.replace(' ', '_')
        if exp_hyphen in pred_norm or exp_underscore in pred_norm:
            return 0.95, "Answer contained (with hyphenation)"
        
        # For short expected answers, check word containment
        exp_words = set(exp_norm.split())
        
        # Find relevant sentence containing potential answer
        # Look for patterns like "is **answer**" or "is: answer"
        answer_patterns = [
            r'is\s+[*]*([^.*\n]+)[*]*',
            r'called\s+[*]*([^.*\n]+)[*]*',
            r':\s*[*]*([^.*\n]+)[*]*',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, pred_norm)
            for match in matches:
                match_clean = match.strip().strip('*').strip()
                # Check if this match overlaps with expected
                if exp_norm in match_clean or match_clean in exp_norm:
                    return 0.9, "Answer extracted from context"
                # Check word overlap
                match_words = set(match_clean.split())
                overlap = len(exp_words & match_words)
                if overlap >= len(exp_words) * 0.8:
                    return 0.85, "High word overlap in extracted answer"
        
        # Fall back to general word overlap
        pred_words = set(pred_norm.split())
        
        if len(exp_words) <= 5:  # Short answer
            overlap = len(exp_words & pred_words) / len(exp_words) if exp_words else 0
            if overlap >= 0.8:
                return 0.8, "High word overlap"
            elif overlap >= 0.6:
                return 0.6, "Moderate word overlap"
            elif overlap >= 0.4:
                return 0.4, "Partial word overlap"
        
        return 0.0, f"Text mismatch: expected '{expected}'"


def create_format_enhanced_prompt(question: str, base_prompt: str = "") -> str:
    """
    Create an enhanced prompt that includes format guidance.
    
    This is the key to improving accuracy: tell the LLM exactly what format
    is expected for the answer.
    """
    analyzer = QuestionAnalyzer()
    prediction = analyzer.analyze(question)
    
    format_guidance = f"""
**IMPORTANT: Answer Format Guidance**
Based on the question type, your final answer should be in this format:
{prediction.format_hint}

Examples of expected format:
{', '.join(prediction.examples[:3]) if prediction.examples else 'N/A'}

Please provide your final answer in this exact format at the end of your response.
"""
    
    if base_prompt:
        return f"{base_prompt}\n\n{format_guidance}\n\nQuestion: {question}"
    else:
        return f"{format_guidance}\n\nQuestion: {question}"


# Test the classifier
if __name__ == "__main__":
    analyzer = QuestionAnalyzer()
    
    test_questions = [
        "If rectangular pulses with duty cycle 1 (NRZ) are used, what is the first-null bandwidth B?",
        "The full name of an m-sequence is ____.",
        "For 2PSK... find the cross-correlation coefficient ρ between s1 and s2.",
        "Write the encoded sequence for input 1101001000111000100001.",
        "State the magnitude spectrum of a raised-cosine pulse.",
        "What is the channel capacity C (in kbit/s)?",
        "What is the RMS delay spread (in seconds)?",
        "Given the FM signal... find the modulating signal f(t).",
    ]
    
    print("=" * 60)
    print("Question Format Analysis")
    print("=" * 60)
    
    for q in test_questions:
        pred = analyzer.analyze(q)
        print(f"\nQ: {q[:70]}...")
        print(f"   Format: {pred.primary_format.value}")
        print(f"   Confidence: {pred.confidence:.2f}")
        print(f"   Unit: {pred.expected_unit}")
        print(f"   Hint: {pred.format_hint[:60]}...")
