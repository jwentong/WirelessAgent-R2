# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class FormatOp(BaseModel):
    solution: str = Field(default="", description="Your formatted answer for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class ReflectionTestOp(BaseModel):
    reflection_and_solution: str = Field(
        default="", description="Corrective solution for code execution errors or test case failures"
    )


class MdEnsembleOp(BaseModel):
    thought: str = Field(default="", description="Step-by-step analysis of the solutions to determine the best one.")
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")


class ReviewOp(BaseModel):
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )


class ReviseOp(BaseModel):
    solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")


class ToolDecisionOp(BaseModel):
    """LLM's decision output format for ReAct Agent
    
    Note: This model allows flexible XML output from LLM.
    All fields have default values to make them truly optional.
    Validation of conditional fields (e.g., tool_name when action_type='use_tool') 
    is handled in ReActAgent logic, not by Pydantic.
    """
    thought: str = Field(default="", description="Current thinking process about what to do next")
    action_type: str = Field(default="final_answer", description="Type of action to take: 'use_tool' or 'final_answer'")
    
    # Optional fields with empty string defaults
    tool_name: str = Field(default="", description="Name of the tool to use (required when action_type='use_tool')")
    tool_args: str = Field(default="", description="Arguments for the tool as JSON string (optional when action_type='use_tool')")
    final_answer: str = Field(default="", description="The final answer to the question (required when action_type='final_answer')")

