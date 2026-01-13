# -*- coding: utf-8 -*-
"""
TF-IDF based Retriever for WCHW Dataset
Retrieves similar problems for few-shot learning
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class WCHWRetriever:
    """
    TF-IDF based retriever for similar problem lookup
    
    Features:
    - Loads problems from JSONL knowledge base
    - TF-IDF vectorization for text similarity
    - Returns top-k most similar problems with reasoning
    """
    
    def __init__(self, knowledge_base_path: str, top_k: int = 3):
        """
        Initialize retriever
        
        Args:
            knowledge_base_path: Path to JSONL file with problems
            top_k: Default number of examples to retrieve
        """
        self.top_k = top_k
        self.problems = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
        self._load_and_index(knowledge_base_path)
    
    def _load_and_index(self, path: str):
        """Load problems and build TF-IDF index"""
        path = Path(path)
        if not path.exists():
            print(f"[RAG] Warning: Knowledge base not found: {path}")
            return
        
        # Load problems
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.problems.append(json.loads(line))
        
        if not self.problems:
            print("[RAG] Warning: No problems loaded")
            return
        
        print(f"[RAG] Loaded {len(self.problems)} problems")
        
        # Build TF-IDF index
        if HAS_SKLEARN:
            texts = [self._extract_text(p) for p in self.problems]
            self.vectorizer = TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            print(f"[RAG] Built TF-IDF index with {self.tfidf_matrix.shape[1]} features")
        else:
            print("[RAG] Warning: sklearn not available, retrieval disabled")
    
    def _extract_text(self, problem: Dict) -> str:
        """Extract searchable text from problem"""
        parts = []
        for key in ['question', 'problem', 'reasoning', 'solution']:
            if key in problem and problem[key]:
                parts.append(str(problem[key]))
        return ' '.join(parts)
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve similar problems
        
        Args:
            query: The problem text to find similar examples for
            top_k: Number of examples to retrieve
            
        Returns:
            Dict with 'similar_problems' list
        """
        top_k = top_k or self.top_k
        
        result = {
            'similar_problems': [],
            'problem_type': 'calculation'
        }
        
        if not HAS_SKLEARN or self.vectorizer is None:
            return result
        
        # Transform query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum threshold
                prob = self.problems[idx].copy()
                prob['similarity'] = float(similarities[idx])
                result['similar_problems'].append(prob)
        
        return result
    
    def format_examples(self, query: str, num_examples: int = 3) -> str:
        """
        Format retrieved examples as few-shot prompt
        
        Args:
            query: Problem to find examples for
            num_examples: Number of examples
            
        Returns:
            Formatted few-shot examples string
        """
        result = self.retrieve(query, top_k=num_examples)
        similar = result.get('similar_problems', [])
        
        if not similar:
            return ""
        
        parts = []
        for i, prob in enumerate(similar, 1):
            question = prob.get('question', prob.get('problem', ''))
            reasoning = prob.get('reasoning', prob.get('solution', ''))
            answer = prob.get('answer', prob.get('ground_truth', ''))
            
            parts.append(f"Example {i}:")
            parts.append(f"Question: {question}")
            if reasoning:
                parts.append(f"Solution: {reasoning}")
            if answer:
                parts.append(f"Answer: {answer}")
            parts.append("")
        
        return '\n'.join(parts)


# Singleton instance
_retriever_instance: Optional[WCHWRetriever] = None


def get_retriever(
    knowledge_base_path: Optional[str] = None,
    enhanced: bool = True  # Ignored, for compatibility
) -> WCHWRetriever:
    """
    Get or create retriever instance (singleton)
    
    Args:
        knowledge_base_path: Path to knowledge base JSONL
        enhanced: Ignored, for API compatibility
        
    Returns:
        WCHWRetriever instance
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        if knowledge_base_path is None:
            # Default path
            base_path = Path(__file__).parent.parent.parent
            knowledge_base_path = base_path / "data" / "datasets" / "wchw_validate.jsonl"
        
        _retriever_instance = WCHWRetriever(
            knowledge_base_path=str(knowledge_base_path),
            top_k=3
        )
    
    return _retriever_instance


def reset_retriever():
    """Reset retriever instance"""
    global _retriever_instance
    _retriever_instance = None
