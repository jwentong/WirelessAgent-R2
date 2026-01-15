"""
Round Test: ScEnsemble Test

Test if ScEnsemble (self-consistency voting) can improve performance.
Generates 3 solutions and votes for the best one.
"""
from typing import Literal
import workspace.WCHW.workflows.template.operator as operator
import workspace.WCHW.workflows.round_7.prompt as prompt_custom
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
        self.tool_agent = operator.ToolAgent(self.llm)
        self.sc_ensemble = operator.ScEnsemble(self.llm)

    async def __call__(self, problem: str):
        """
        ScEnsemble workflow: Generate 3 solutions and vote for the best one.
        
        Strategy:
        1. Generate 3 independent solutions using Custom + ToolAgent
        2. Use ScEnsemble to vote and select the most consistent answer
        """
        solutions = []
        
        # Generate 3 independent solutions
        for i in range(3):
            # Step 1: LLM analyzes problem
            analysis = await self.custom(
                input=problem, 
                instruction=prompt_custom.ANALYZE_AND_SOLVE_PROMPT
            )
            
            # Step 2: ToolAgent executes the calculation
            result = await self.tool_agent(
                problem=f"Problem: {problem}\n\nSolution approach: {analysis['response']}\n\nExecute the calculation and output ONLY the final numerical answer as a pure number in base units (Hz not kHz, W not mW, s not ms, bit/s not kbit/s, nats not bits). No units, no text, no explanation. Just the number.",
                max_steps=3
            )
            solutions.append(str(result['answer']))
        
        # Step 3: Vote using ScEnsemble
        try:
            final = await self.sc_ensemble(solutions=solutions, problem=problem)
            return final['response'], self.llm.get_usage_summary()["total_cost"]
        except Exception:
            # Fallback: return first solution if voting fails
            return solutions[0], self.llm.get_usage_summary()["total_cost"]
