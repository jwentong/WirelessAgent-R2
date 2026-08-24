# -*- coding: utf-8 -*-
# @Date    : Feb.10
# @Author  : Jwen
# @Desc    : WCNS (Network Slicing) Agent - Output Schemas

from pydantic import BaseModel, Field
from typing import Optional, List


class GenerateOp(BaseModel):
    """General response output schema"""
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    """Python code generation output schema"""
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    """Self-consistency ensemble selection output"""
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class IntentAnalysisOp(BaseModel):
    """Intent understanding output schema"""
    service_type: str = Field(default="", description="Identified service type (e.g., video_streaming, online_gaming)")
    slice_recommendation: str = Field(default="eMBB", description="Recommended slice type: eMBB or URLLC")
    min_rate_mbps: float = Field(default=0.0, description="Minimum required rate in Mbps")
    reasoning: str = Field(default="", description="Explanation of intent analysis")


class BandwidthAllocationOp(BaseModel):
    """Bandwidth allocation calculation output schema"""
    slice_type: str = Field(default="eMBB", description="Allocated slice type: eMBB or URLLC")
    allocated_bandwidth_mhz: float = Field(default=0.0, description="Allocated bandwidth in MHz")
    expected_rate_mbps: float = Field(default=0.0, description="Expected rate in Mbps based on Shannon formula")
    calculation_steps: str = Field(default="", description="Step-by-step calculation explanation")


class QoSEvaluationOp(BaseModel):
    """QoS evaluation output schema"""
    qos_satisfied: bool = Field(default=False, description="Whether QoS requirements are satisfied")
    rate_check: bool = Field(default=False, description="Rate >= min_rate check result")
    bandwidth_check: bool = Field(default=False, description="Bandwidth within constraints check result")
    recommendations: str = Field(default="", description="Recommendations if QoS not satisfied")


class SliceAllocationResult(BaseModel):
    """Final slice allocation result schema"""
    user_id: str = Field(default="", description="User identifier")
    service_type: str = Field(default="", description="Detected service type")
    slice_type: str = Field(default="eMBB", description="Final allocated slice: eMBB or URLLC")
    allocated_bandwidth: float = Field(default=0.0, description="Allocated bandwidth in MHz")
    expected_rate: float = Field(default=0.0, description="Expected throughput rate in Mbps")
    qos_satisfied: bool = Field(default=True, description="Whether QoS requirements are met")


class NetworkSlicingDecision(BaseModel):
    """Complete network slicing decision output"""
    slice_type: str = Field(default="eMBB", description="Selected slice: eMBB or URLLC")
    bandwidth: float = Field(default=10.0, description="Allocated bandwidth in MHz")
    rate: float = Field(default=100.0, description="Expected rate in Mbps")
    reasoning: str = Field(default="", description="Decision reasoning")


class ToolCallResult(BaseModel):
    """Tool execution result schema"""
    tool_name: str = Field(default="", description="Name of the tool called")
    success: bool = Field(default=True, description="Whether tool execution succeeded")
    result: str = Field(default="", description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ReActStep(BaseModel):
    """ReAct reasoning step schema"""
    thought: str = Field(default="", description="Current reasoning thought")
    action: str = Field(default="", description="Action to take (tool name or 'final_answer')")
    action_input: str = Field(default="", description="Input for the action")
    observation: Optional[str] = Field(default=None, description="Result from action execution")


class ReActOutput(BaseModel):
    """Complete ReAct process output"""
    steps: List[ReActStep] = Field(default_factory=list, description="List of reasoning steps")
    final_answer: str = Field(default="", description="Final answer after reasoning")
    slice_type: str = Field(default="eMBB", description="Determined slice type")
    allocated_bandwidth: float = Field(default=10.0, description="Allocated bandwidth in MHz")
    expected_rate: float = Field(default=100.0, description="Expected rate in Mbps")
