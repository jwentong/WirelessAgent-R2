# -*- coding: utf-8 -*-
"""
RAG Tool Module for Telecommunications Knowledge Retrieval

This module provides a RAG (Retrieval Augmented Generation) tool that can be
used by ToolAgent to retrieve relevant formulas and domain knowledge.

The RAG tool is designed to work alongside the code execution tool,
allowing the optimizer to choose the best approach for each problem.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class RetrievedFormula:
    """A retrieved formula with relevance score."""
    id: str
    name: str
    formula_text: str
    formula_latex: str
    category: str
    relevance_score: float
    keywords_matched: List[str]
    variables: Dict[str, str]
    notes: str
    calculator_function: Optional[str] = None


class TelecomRAGTool:
    """
    RAG tool for retrieving telecommunications formulas and knowledge.
    
    This tool searches a structured knowledge base to find relevant
    formulas based on the question content.
    """
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        Initialize the RAG tool.
        
        Args:
            knowledge_base_path: Path to the knowledge base JSON file.
                               If None, uses the default location.
        """
        if knowledge_base_path is None:
            # Default path relative to this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            knowledge_base_path = os.path.join(current_dir, 'telecom_knowledge_base.json')
        
        self.knowledge_base_path = knowledge_base_path
        self.formulas: List[Dict] = []
        self._load_knowledge_base()
        
        # Build keyword index for fast retrieval
        self._build_index()
    
    def _load_knowledge_base(self):
        """Load the knowledge base from JSON file."""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.formulas = data.get('formulas', [])
        except FileNotFoundError:
            print(f"Warning: Knowledge base not found at {self.knowledge_base_path}")
            self.formulas = []
    
    def _build_index(self):
        """Build inverted index for keyword search."""
        self.keyword_index: Dict[str, List[int]] = {}
        
        for idx, formula in enumerate(self.formulas):
            # Index by keywords
            for keyword in formula.get('keywords', []):
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_index:
                    self.keyword_index[keyword_lower] = []
                self.keyword_index[keyword_lower].append(idx)
            
            # Also index by category and name
            category = formula.get('category', '').lower()
            if category:
                if category not in self.keyword_index:
                    self.keyword_index[category] = []
                self.keyword_index[category].append(idx)
            
            # Index name words
            name_words = formula.get('name', '').lower().split()
            for word in name_words:
                if len(word) > 2:  # Skip short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(idx)
    
    def _extract_query_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from a query."""
        # Common telecom terms to look for
        important_terms = [
            'ber', 'bpsk', 'qpsk', 'qam', 'fsk', 'psk', 'dpsk',
            'capacity', 'shannon', 'snr', 'noise',
            'rayleigh', 'rician', 'fading', 'outage',
            'bessel', 'erfc', 'erf', 'marcum',
            'doppler', 'coherence',
            'fm', 'am', 'modulation',
            'diversity', 'mrc', 'combining',
            'path loss', 'propagation',
            'error rate', 'bit error',
            # Chinese terms
            '误码率', '信道容量', '衰落', '调制', '噪声', '分集'
        ]
        
        query_lower = query.lower()
        found_terms = []
        
        # Check for multi-word terms first
        for term in sorted(important_terms, key=len, reverse=True):
            if term in query_lower:
                found_terms.append(term)
        
        # Also split query into words and check single-word matches
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]+\b', query_lower)
        for word in words:
            if len(word) > 2 and word in self.keyword_index:
                if word not in found_terms:
                    found_terms.append(word)
        
        return found_terms
    
    def _calculate_relevance(self, formula: Dict, query_keywords: List[str], query: str) -> Tuple[float, List[str]]:
        """Calculate relevance score for a formula given query keywords."""
        score = 0.0
        matched_keywords = []
        
        formula_keywords = [k.lower() for k in formula.get('keywords', [])]
        formula_name = formula.get('name', '').lower()
        formula_notes = formula.get('notes', '').lower()
        formula_text = formula.get('formula_text', '').lower()
        
        query_lower = query.lower()
        
        for qk in query_keywords:
            # Direct keyword match (highest weight)
            if qk in formula_keywords:
                score += 2.0
                matched_keywords.append(qk)
            
            # Partial match in name
            elif qk in formula_name:
                score += 1.5
                matched_keywords.append(qk)
            
            # Match in notes
            elif qk in formula_notes:
                score += 0.5
                matched_keywords.append(qk)
            
            # Match in formula text
            elif qk in formula_text:
                score += 0.3
                matched_keywords.append(qk)
        
        # Bonus for category match
        category = formula.get('category', '').lower()
        if category and category in query_lower:
            score += 1.0
        
        return score, matched_keywords
    
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievedFormula]:
        """
        Retrieve relevant formulas for a query.
        
        Args:
            query: The question or query text
            top_k: Number of top results to return
            
        Returns:
            List of RetrievedFormula objects sorted by relevance
        """
        if not self.formulas:
            return []
        
        query_keywords = self._extract_query_keywords(query)
        
        # Calculate scores for all formulas
        scored_formulas = []
        for formula in self.formulas:
            score, matched = self._calculate_relevance(formula, query_keywords, query)
            if score > 0:
                scored_formulas.append((score, matched, formula))
        
        # Sort by score descending
        scored_formulas.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to RetrievedFormula objects
        results = []
        for score, matched, formula in scored_formulas[:top_k]:
            results.append(RetrievedFormula(
                id=formula.get('id', ''),
                name=formula.get('name', ''),
                formula_text=formula.get('formula_text', ''),
                formula_latex=formula.get('formula_latex', ''),
                category=formula.get('category', ''),
                relevance_score=score,
                keywords_matched=matched,
                variables=formula.get('variables', {}),
                notes=formula.get('notes', ''),
                calculator_function=formula.get('calculator_function')
            ))
        
        return results
    
    def format_retrieval_result(self, results: List[RetrievedFormula]) -> str:
        """Format retrieval results as a string for LLM context."""
        if not results:
            return "No relevant formulas found in knowledge base."
        
        output_parts = ["### Retrieved Formulas from Knowledge Base:\n"]
        
        for i, r in enumerate(results, 1):
            output_parts.append(f"\n**Formula {i}: {r.name}** (relevance: {r.relevance_score:.1f})")
            output_parts.append(f"- Category: {r.category}")
            output_parts.append(f"- Formula: `{r.formula_text}`")
            if r.formula_latex:
                output_parts.append(f"- LaTeX: ${r.formula_latex}$")
            if r.variables:
                vars_str = ", ".join([f"{k}={v}" for k, v in r.variables.items()])
                output_parts.append(f"- Variables: {vars_str}")
            if r.notes:
                output_parts.append(f"- Notes: {r.notes}")
            if r.calculator_function:
                output_parts.append(f"- Calculator: `{r.calculator_function}`")
            output_parts.append("")
        
        return "\n".join(output_parts)


class RAGToolWrapper:
    """
    Wrapper class that provides a tool interface compatible with ToolAgent.
    
    This allows the RAG tool to be used alongside code execution in ToolAgent.
    """
    
    TOOL_DESCRIPTION = """
formula_retrieval: Retrieve relevant telecommunications formulas and domain knowledge.
Use this tool when:
- You need to find the correct formula for a specific calculation
- You're unsure about the exact mathematical relationship
- You need to know variable definitions or units
- You want to verify your approach with standard formulas

Input: A description of what you're looking for (e.g., "BPSK BER formula", "Rayleigh fading outage probability")
Output: Relevant formulas with explanations and calculator references
"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        self.rag = TelecomRAGTool(knowledge_base_path)
    
    def __call__(self, query: str) -> Dict[str, Any]:
        """
        Execute the RAG retrieval.
        
        Args:
            query: What to search for
            
        Returns:
            Dict with retrieval results
        """
        results = self.rag.retrieve(query, top_k=3)
        formatted = self.rag.format_retrieval_result(results)
        
        return {
            'success': True,
            'num_results': len(results),
            'formatted_output': formatted,
            'raw_results': [
                {
                    'id': r.id,
                    'name': r.name,
                    'formula_text': r.formula_text,
                    'category': r.category,
                    'calculator_function': r.calculator_function
                }
                for r in results
            ]
        }
    
    @property
    def tool_spec(self) -> Dict[str, Any]:
        """Return tool specification for optimizer."""
        return {
            'name': 'formula_retrieval',
            'description': self.TOOL_DESCRIPTION,
            'parameters': {
                'query': {
                    'type': 'string',
                    'description': 'Description of the formula or knowledge to retrieve'
                }
            },
            'required': ['query']
        }


# Convenience function
def retrieve_formulas(query: str, top_k: int = 3) -> str:
    """
    Simple function to retrieve formulas.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        Formatted string with results
    """
    rag = TelecomRAGTool()
    results = rag.retrieve(query, top_k)
    return rag.format_retrieval_result(results)


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Tool Test")
    print("=" * 60)
    
    rag = TelecomRAGTool()
    
    # Test queries
    test_queries = [
        "What is the BER for BPSK modulation?",
        "Calculate Shannon channel capacity",
        "Rayleigh fading outage probability",
        "如何计算误码率",
        "Marcum Q function for Rician fading"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: {query}")
        print('='*40)
        
        results = rag.retrieve(query, top_k=2)
        print(rag.format_retrieval_result(results))
    
    # Test wrapper
    print("\n" + "=" * 60)
    print("Testing RAGToolWrapper...")
    wrapper = RAGToolWrapper()
    result = wrapper("BPSK BER coherent detection")
    print(f"Success: {result['success']}")
    print(f"Num results: {result['num_results']}")
    print(result['formatted_output'])
