# -*- coding: utf-8 -*-
"""
WCMSA (Wireless Communication Mobile Service Assurance) Workspace

This workspace contains operators, tools, and workflows for proactive resource
allocation based on Kalman Filter trajectory prediction.

Key Features:
- Kalman Filter for 2D trajectory prediction (state: [x, y, vx, vy])
- Path loss model for CQI calculation from predicted position
- Shannon formula for bandwidth allocation
- eMBB/URLLC slice selection

Main Components:
- workflows/template/operator.py: Core operators (MobileServiceAssuranceAgent, etc.)
- workflows/template/op_prompt.py: Prompts and WCMSA knowledge base
- workflows/template/operator_an.py: Output schemas (Pydantic models)
- workflows/template/operator.json: Operator metadata

Dataset: data/datasets/wcmsa_validate.jsonl, wcmsa_test.jsonl
Benchmark: benchmarks/wcmsa.py (WCMSABenchmark, WCMSABenchmarkWithCoT)

Usage:
    from workspace.WCMSA.workflows.template import MobileServiceAssuranceAgent
    from scripts.async_llm import AsyncLLM
    
    llm = AsyncLLM(...)
    agent = MobileServiceAssuranceAgent(llm)
    result = await agent(
        problem="User requests video streaming service",
        trajectory_history=[(0, 0), (5, 3), (10, 6), (15, 9)],
        bs_position=[0, 0]
    )
    
    # Result contains:
    # - predicted_position: (20.0, 12.0)
    # - predicted_cqi: 10
    # - slice_type: "eMBB"
    # - bandwidth: 12.5 MHz
    # - rate: 156.8 Mbps
"""

__version__ = "1.0.0"
__author__ = "Jwen"
