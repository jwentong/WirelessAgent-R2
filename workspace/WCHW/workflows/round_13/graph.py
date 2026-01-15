from typing import Literal
import workspace.WCHW.workflows.template.operator as operator
import workspace.WCHW.workflows.round_13.prompt as prompt_custom
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

    async def __call__(self, problem: str):
        """
        Implementation of the workflow with calculation verification
        """
        # Step 1: LLM solves the problem with reasoning
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        
        # Step 2: ToolAgent verifies and extracts final numerical answer
        verification = await self.tool_agent(
            problem=f"Problem: {problem}\n\nProposed solution: {solution['response']}\n\nVerify the calculation and output ONLY the final numerical answer as a pure number in base units (Hz not kHz, W not mW, s not ms). No units, no text, just the number.",
            max_steps=2
        )
        
        return verification['answer'], self.llm.get_usage_summary()["total_cost"]
