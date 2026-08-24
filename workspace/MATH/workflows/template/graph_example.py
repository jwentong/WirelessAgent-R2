"""
Template workflow for MATH dataset

This file serves as a reference example for the optimizer.
The optimizer will generate actual graph.py files in round_X directories.

Key features:
1. ToolAgent now supports optional react_strategy parameter
2. Strategy can be optimized via prompt_custom.REACT_STRATEGY_PROMPT
"""

from typing import Literal
import workspace.MATH.workflows.template.operator as operator
# Note: prompt_custom will be from the specific round, e.g.:
# import workspace.MATH.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType


class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        
        # Default: Use built-in REACT_STRATEGY_PROMPT from op_prompt.py
        self.tool_agent = operator.ToolAgent(self.llm)
        
        # Advanced: Pass custom strategy from prompt_custom (for optimizer to modify)
        # Uncomment and use this pattern when you want to optimize the ReAct strategy:
        # self.tool_agent = operator.ToolAgent(
        #     self.llm, 
        #     react_strategy=prompt_custom.REACT_STRATEGY_PROMPT
        # )

    async def __call__(self, problem: str):
        """
        Simple baseline workflow
        """
        # Use ToolAgent for computational problem solving
        result = await self.tool_agent(problem=problem, max_steps=5)
        
        return result['answer'], self.llm.get_usage_summary()["total_cost"]
