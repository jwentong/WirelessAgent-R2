# -*- coding: utf-8 -*-
"""
WCNS (Wireless Communication Network Slicing) Agent - Template Module

This module provides operators and tools for network slicing decision-making:

Operators:
- Custom: Basic instruction-based operator
- ScEnsemble: Self-consistency voting for higher accuracy
- NetworkSlicingAgent: ReAct-based agent with domain-specific tools
- DirectSlicingSolver: Fast direct calculation without ReAct

Tools (used by NetworkSlicingAgent):
- knowledge_base_query: Query slicing knowledge base
- shannon_calculator: Calculate rate from CQI and bandwidth
- slice_allocator: Allocate resources to slice
- network_state_checker: Check network utilization
- optimal_bandwidth_calculator: Calculate minimum bandwidth for QoS

Workflow:
1. Intent Understanding → Identify service type from user request
2. Slice Selection → eMBB (bandwidth) vs URLLC (latency)
3. Bandwidth Calculation → Shannon formula with CQI
4. QoS Verification → Check min rate requirements
5. Final Allocation → slice_type, bandwidth, rate
"""

from workspace.WCNS.workflows.template.operator import (
    Custom,
    ScEnsemble,
    NetworkSlicingAgent,
    DirectSlicingSolver,
    # Tools
    KnowledgeBaseQueryTool,
    ShannonCalculatorTool,
    SliceAllocatorTool,
    NetworkStateCheckerTool,
    OptimalBandwidthCalculatorTool
)

from workspace.WCNS.workflows.template.operator_an import (
    GenerateOp,
    CodeGenerateOp,
    ScEnsembleOp,
    IntentAnalysisOp,
    BandwidthAllocationOp,
    QoSEvaluationOp,
    SliceAllocationResult,
    NetworkSlicingDecision,
    ReActOutput
)

from workspace.WCNS.workflows.template.op_prompt import (
    SC_ENSEMBLE_PROMPT,
    SLICE_KNOWLEDGE_BASE,
    INTENT_UNDERSTANDING_PROMPT,
    BANDWIDTH_ALLOCATION_PROMPT,
    QOS_EVALUATION_PROMPT,
    REACT_STRATEGY_PROMPT,
    PYTHON_CODE_SOLVER_PROMPT,
    VERIFY_ALLOCATION_PROMPT
)

__all__ = [
    # Operators
    "Custom",
    "ScEnsemble", 
    "NetworkSlicingAgent",
    "DirectSlicingSolver",
    # Tools
    "KnowledgeBaseQueryTool",
    "ShannonCalculatorTool",
    "SliceAllocatorTool",
    "NetworkStateCheckerTool",
    "OptimalBandwidthCalculatorTool",
    # Output Schemas
    "GenerateOp",
    "ScEnsembleOp",
    "IntentAnalysisOp",
    "BandwidthAllocationOp",
    "NetworkSlicingDecision",
    # Prompts
    "SC_ENSEMBLE_PROMPT",
    "REACT_STRATEGY_PROMPT",
    "SLICE_KNOWLEDGE_BASE"
]
