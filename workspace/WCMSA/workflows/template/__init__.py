# -*- coding: utf-8 -*-
"""
WCMSA (Wireless Communication Mobile Service Assurance) Agent - Template Module

This module provides operators and tools for proactive resource allocation
with Kalman Filter trajectory prediction:

Operators:
- Custom: Basic instruction-based operator
- ScEnsemble: Self-consistency voting for higher accuracy
- MobileServiceAssuranceAgent: ReAct-based agent with Kalman prediction
- DirectMobileSolver: Fast direct calculation without ReAct

Tools (used by MobileServiceAssuranceAgent):
- kalman_filter_predictor: Predict next position from trajectory history
- cqi_calculator: Calculate CQI at a position using path loss model
- shannon_calculator: Calculate rate from CQI and bandwidth
- slice_allocator: Allocate resources to network slice
- knowledge_base_query: Query service requirements and parameters

Workflow (6 steps):
1. Historical Trajectory Analysis → Extract motion pattern
2. Kalman Filter Prediction → Predict next position
3. CQI Calculation → Channel quality at predicted position
4. Slice Selection → eMBB vs URLLC
5. Bandwidth Allocation → Shannon formula with predicted CQI
6. QoS Verification → Verify rate meets requirements
"""

from workspace.WCMSA.workflows.template.operator import (
    # Operators
    Custom,
    ScEnsemble,
    MobileServiceAssuranceAgent,
    DirectMobileSolver,
    # Tools
    KalmanFilterPredictorTool,
    CQICalculatorTool,
    ShannonCalculatorTool,
    SliceAllocatorTool,
    KnowledgeBaseQueryTool
)

from workspace.WCMSA.workflows.template.operator_an import (
    GenerateOp,
    CodeGenerateOp,
    ScEnsembleOp,
    TrajectoryPoint,
    KalmanState,
    KalmanPredictionOp,
    CQICalculationOp,
    BandwidthAllocationOp,
    QoSEvaluationOp,
    MobileServiceDecision,
    ProactiveAllocationResult,
    ReActOutput
)

from workspace.WCMSA.workflows.template.op_prompt import (
    SC_ENSEMBLE_PROMPT,
    WCMSA_KNOWLEDGE_BASE,
    KALMAN_PREDICTION_PROMPT,
    CQI_CALCULATION_PROMPT,
    REACT_STRATEGY_PROMPT,
    PYTHON_CODE_SOLVER_PROMPT,
    VERIFY_ALLOCATION_PROMPT,
    INTENT_UNDERSTANDING_PROMPT
)

__all__ = [
    # Operators
    "Custom",
    "ScEnsemble", 
    "MobileServiceAssuranceAgent",
    "DirectMobileSolver",
    # Tools
    "KalmanFilterPredictorTool",
    "CQICalculatorTool",
    "ShannonCalculatorTool",
    "SliceAllocatorTool",
    "KnowledgeBaseQueryTool",
    # Output Schemas
    "GenerateOp",
    "ScEnsembleOp",
    "KalmanPredictionOp",
    "CQICalculationOp",
    "MobileServiceDecision",
    "ProactiveAllocationResult",
    # Prompts
    "SC_ENSEMBLE_PROMPT",
    "REACT_STRATEGY_PROMPT",
    "WCMSA_KNOWLEDGE_BASE"
]
