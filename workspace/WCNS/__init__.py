# -*- coding: utf-8 -*-
"""
WCNS (Wireless Communication Network Slicing) Workspace

This workspace contains operators, tools, and workflows for 5G network slicing
decision-making tasks.

Main Components:
- workflows/template/operator.py: Core operators (NetworkSlicingAgent, etc.)
- workflows/template/op_prompt.py: Prompts and knowledge base
- workflows/template/operator_an.py: Output schemas (Pydantic models)
- workflows/template/operator.json: Operator metadata

Dataset: data/datasets/wcns_validate.jsonl, wcns_test.jsonl
Benchmark: benchmarks/wcns.py (WCNSBenchmark, WCNSBenchmarkWithCoT)

Usage:
    from workspace.WCNS.workflows.template import NetworkSlicingAgent
    from scripts.async_llm import AsyncLLM
    
    llm = AsyncLLM(...)
    agent = NetworkSlicingAgent(llm)
    result = await agent(
        problem="User requests video streaming service",
        user_cqi=12
    )
"""

__version__ = "1.0.0"
__author__ = "Jwen"
