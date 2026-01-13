"""
WirelessAgent Green Agent - Agent Implementation
UC Berkeley AgentX Competition - AgentBeats Compatible

This module contains the core WCHW benchmark agent logic.

Author: Jingwen
Date: 1/13/2026
"""

import json
import logging
import os
import re
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("WirelessAgent.Agent")

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class WCHWEvaluator:
    """
    Evaluator for WCHW Benchmark Answers
    
    Supports multiple answer types:
    - Numeric with units (e.g., "16 kbit/s", "44.8 kHz")
    - Scientific notation (e.g., "5.42e-6")
    - Mathematical formulas (e.g., "(A^2 T)/3")
    - Text/conceptual answers
    """
    
    # Unit conversion factors
    UNIT_PREFIXES = {
        'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3,
        'm': 1e-3, 'μ': 1e-6, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12
    }
    
    def evaluate(self, question: str, prediction: str, answer: str) -> float:
        """
        Evaluate a prediction against the ground truth answer.
        
        Returns:
            float: Score between 0.0 and 1.0
        """
        if not prediction or not answer:
            return 0.0
        
        # Normalize strings
        pred_clean = self._normalize(prediction)
        ans_clean = self._normalize(answer)
        
        # Exact match
        if pred_clean == ans_clean:
            return 1.0
        
        # Try numeric comparison
        pred_value = self._extract_numeric(prediction)
        ans_value = self._extract_numeric(answer)
        
        if pred_value is not None and ans_value is not None:
            return self._score_numeric(pred_value, ans_value)
        
        # Try formula comparison
        if self._is_formula(answer):
            return self._score_formula(prediction, answer)
        
        # Fallback to text similarity
        return self._score_text(prediction, answer)
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.strip().lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('×', '*').replace('·', '*')
        return text
    
    def _extract_numeric(self, text: str) -> Optional[float]:
        """Extract numeric value with unit handling"""
        text = text.strip()
        
        # Scientific notation: 5.42e-6, 2.2×10^-8
        sci_match = re.search(r'([+-]?\d+\.?\d*)\s*[×xX*]?\s*10\s*\^?\s*([+-]?\d+)', text)
        if sci_match:
            mantissa = float(sci_match.group(1))
            exponent = int(sci_match.group(2))
            return mantissa * (10 ** exponent)
        
        # Standard scientific: 5.42e-6
        sci_match = re.search(r'([+-]?\d+\.?\d*)[eE]([+-]?\d+)', text)
        if sci_match:
            return float(sci_match.group(0))
        
        # Number with unit prefix: 16 kbit/s, 44.8 MHz
        unit_match = re.search(r'([+-]?\d+\.?\d*)\s*([TGMkmuμnp])?', text)
        if unit_match:
            value = float(unit_match.group(1))
            prefix = unit_match.group(2)
            if prefix and prefix in self.UNIT_PREFIXES:
                value *= self.UNIT_PREFIXES[prefix]
            return value
        
        # Plain number
        num_match = re.search(r'[+-]?\d+\.?\d*', text)
        if num_match:
            return float(num_match.group(0))
        
        return None
    
    def _score_numeric(self, pred: float, ans: float) -> float:
        """Score numeric prediction using relative error"""
        if ans == 0:
            return 1.0 if pred == 0 else 0.0
        
        rel_error = abs(pred - ans) / abs(ans)
        
        if rel_error < 0.01:  # < 1%
            return 1.0
        elif rel_error < 0.05:  # < 5%
            return 0.9
        elif rel_error < 0.10:  # < 10%
            return 0.7
        elif rel_error < 0.20:  # < 20%
            return 0.5
        else:
            return 0.0
    
    def _is_formula(self, text: str) -> bool:
        """Check if text is a mathematical formula"""
        formula_chars = ['^', '/', '*', '(', ')', '\\', 'log', 'sin', 'cos', 'exp']
        return any(c in text for c in formula_chars)
    
    def _score_formula(self, pred: str, ans: str) -> float:
        """Score formula by normalizing and comparing"""
        # Normalize formulas
        pred_norm = self._normalize_formula(pred)
        ans_norm = self._normalize_formula(ans)
        
        if pred_norm == ans_norm:
            return 1.0
        
        # Check if key components match
        pred_tokens = set(re.findall(r'[a-zA-Z_]\w*|\d+\.?\d*', pred_norm))
        ans_tokens = set(re.findall(r'[a-zA-Z_]\w*|\d+\.?\d*', ans_norm))
        
        if not ans_tokens:
            return 0.0
        
        overlap = len(pred_tokens & ans_tokens) / len(ans_tokens)
        return 0.8 if overlap > 0.8 else overlap * 0.5
    
    def _normalize_formula(self, text: str) -> str:
        """Normalize mathematical formula"""
        text = text.strip()
        text = re.sub(r'\s+', '', text)
        text = text.replace('×', '*').replace('·', '*')
        text = text.replace('{', '(').replace('}', ')')
        text = text.replace('[', '(').replace(']', ')')
        text = re.sub(r'\\[a-z]+', '', text)  # Remove LaTeX commands
        return text.lower()
    
    def _score_text(self, pred: str, ans: str) -> float:
        """Score text answer by keyword matching"""
        pred_words = set(self._normalize(pred).split())
        ans_words = set(self._normalize(ans).split())
        
        if not ans_words:
            return 0.0
        
        overlap = len(pred_words & ans_words) / len(ans_words)
        return 1.0 if overlap > 0.8 else overlap


class WCHWAgent:
    """
    WCHW Benchmark Agent
    
    Loads the test dataset and provides evaluation capabilities.
    
    Supports two modes:
    - "test": 100 test problems for final evaluation
    - "validate": 349 validation problems for development
    """
    
    # Dataset paths by mode
    DATASETS = {
        "test": "data/datasets/wchw_test.jsonl",
        "validate": "data/datasets/wchw_validate.jsonl"
    }
    
    def __init__(self, dataset_path: str = None, mode: str = None):
        """
        Initialize WCHW Agent.
        
        Args:
            dataset_path: Direct path to dataset file (overrides mode)
            mode: "test" or "validate" (default: from ASSESSMENT_MODE env var or "test")
        """
        # Determine mode from environment variable or parameter
        if mode is None:
            mode = os.environ.get("ASSESSMENT_MODE", "test")
        
        self.mode = mode
        
        if dataset_path is None:
            # Use mode-based path
            dataset_file = self.DATASETS.get(mode, self.DATASETS["test"])
            dataset_path = PROJECT_ROOT / dataset_file
        
        self.dataset_path = Path(dataset_path)
        self.evaluator = WCHWEvaluator()
        self.problems: List[Dict] = []
        self.results: Dict[str, Dict] = {}
        
        self._load_dataset()
        logger.info(f"WCHWAgent initialized in '{self.mode}' mode with {len(self.problems)} problems")
    
    def _load_dataset(self):
        """Load WCHW test problems"""
        try:
            if self.dataset_path.exists():
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    self.problems = [json.loads(line) for line in f if line.strip()]
                logger.info(f"Loaded {len(self.problems)} problems from {self.dataset_path}")
            else:
                logger.warning(f"Dataset not found: {self.dataset_path}")
                self.problems = []
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.problems = []
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Get all assessment tasks"""
        return [
            {
                "task_id": p.get("id", f"task_{i}"),
                "question": p["question"],
                "metadata": {
                    "index": i,
                    "total": len(self.problems)
                }
            }
            for i, p in enumerate(self.problems)
        ]
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID"""
        for i, p in enumerate(self.problems):
            if p.get("id", f"task_{i}") == task_id:
                return {
                    "task_id": task_id,
                    "question": p["question"],
                    "metadata": {"index": i}
                }
        return None
    
    def evaluate_response(self, task_id: str, response: str) -> Dict[str, Any]:
        """Evaluate a response for a given task"""
        # Find the problem
        problem = None
        for p in self.problems:
            if p.get("id", "") == task_id:
                problem = p
                break
        
        if problem is None:
            # Try matching by index
            for i, p in enumerate(self.problems):
                if f"task_{i}" == task_id or f"test_{i}" == task_id:
                    problem = p
                    break
        
        if problem is None:
            return {
                "task_id": task_id,
                "status": "error",
                "error": "Task not found",
                "score": 0.0
            }
        
        # Evaluate
        score = self.evaluator.evaluate(
            question=problem["question"],
            prediction=response,
            answer=problem["answer"]
        )
        
        result = {
            "task_id": task_id,
            "status": "success",
            "score": score,
            "max_score": 1.0,
            "passed": score >= 0.5,
            "details": {
                "prediction": response,
                "ground_truth": problem["answer"]
            }
        }
        
        self.results[task_id] = result
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get assessment summary"""
        if not self.results:
            return {
                "total_tasks": len(self.problems),
                "completed": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "time_used": 0.0
            }
        
        scores = [r["score"] for r in self.results.values()]
        passed = sum(1 for r in self.results.values() if r.get("passed", False))
        
        return {
            "total_tasks": len(self.problems),
            "completed": len(self.results),
            "pass_rate": passed / len(self.results) if self.results else 0,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "time_used": 0.0,  # Placeholder for actual timing
            "results": self.results
        }
    
    def reset(self):
        """Reset evaluation results"""
        self.results = {}
