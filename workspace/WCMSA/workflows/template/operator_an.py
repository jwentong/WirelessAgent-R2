# -*- coding: utf-8 -*-
# @Date    : Feb.10
# @Author  : Jwen
# @Desc    : WCMSA (Mobile Service Assurance) Agent - Output Schemas

from pydantic import BaseModel, Field
from typing import Optional, List, Tuple


class GenerateOp(BaseModel):
    """General response output schema"""
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    """Python code generation output schema"""
    code: str = Field(default="", description="Your complete code solution for this problem")


class ScEnsembleOp(BaseModel):
    """Self-consistency ensemble selection output"""
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")


class TrajectoryPoint(BaseModel):
    """Single point in trajectory"""
    x: float = Field(description="X coordinate in meters")
    y: float = Field(description="Y coordinate in meters")
    timestamp: Optional[float] = Field(default=None, description="Time in seconds")


class KalmanState(BaseModel):
    """Kalman Filter state output schema"""
    position_x: float = Field(description="Predicted X position")
    position_y: float = Field(description="Predicted Y position")
    velocity_x: float = Field(description="Estimated X velocity (m/s)")
    velocity_y: float = Field(description="Estimated Y velocity (m/s)")
    uncertainty: float = Field(default=0.0, description="Prediction uncertainty (covariance trace)")


class KalmanPredictionOp(BaseModel):
    """Kalman Filter prediction output schema"""
    predicted_position: Tuple[float, float] = Field(description="Predicted (x, y) position")
    predicted_velocity: Tuple[float, float] = Field(description="Estimated (vx, vy) velocity")
    prediction_steps: str = Field(default="", description="Step-by-step prediction explanation")
    confidence: float = Field(default=0.9, description="Prediction confidence (0-1)")


class CQICalculationOp(BaseModel):
    """CQI calculation output schema"""
    position: Tuple[float, float] = Field(description="User position (x, y)")
    distance_to_bs: float = Field(description="Distance to base station in meters")
    path_loss_db: float = Field(description="Path loss in dB")
    snr_db: float = Field(description="SNR in dB")
    cqi: int = Field(description="Channel Quality Indicator (1-15)")
    calculation_steps: str = Field(default="", description="Step-by-step calculation")


class BandwidthAllocationOp(BaseModel):
    """Bandwidth allocation calculation output schema"""
    slice_type: str = Field(default="eMBB", description="Allocated slice type: eMBB or URLLC")
    allocated_bandwidth_mhz: float = Field(default=0.0, description="Allocated bandwidth in MHz")
    expected_rate_mbps: float = Field(default=0.0, description="Expected rate in Mbps")
    calculation_steps: str = Field(default="", description="Step-by-step calculation explanation")


class QoSEvaluationOp(BaseModel):
    """QoS evaluation output schema"""
    qos_satisfied: bool = Field(default=False, description="Whether QoS requirements are satisfied")
    rate_check: bool = Field(default=False, description="Rate >= min_rate check result")
    bandwidth_check: bool = Field(default=False, description="Bandwidth within constraints check")
    latency_check: bool = Field(default=True, description="Latency within limits check")
    recommendations: str = Field(default="", description="Recommendations if QoS not satisfied")


class MobileServiceDecision(BaseModel):
    """Complete mobile service assurance decision output"""
    # Prediction results
    predicted_position_x: float = Field(description="Predicted X position")
    predicted_position_y: float = Field(description="Predicted Y position")
    predicted_cqi: int = Field(default=10, description="Predicted CQI at next position")
    
    # Allocation results
    slice_type: str = Field(default="eMBB", description="Selected slice: eMBB or URLLC")
    bandwidth: float = Field(default=10.0, description="Allocated bandwidth in MHz")
    rate: float = Field(default=100.0, description="Expected rate in Mbps")
    
    # Metadata
    service_type: str = Field(default="", description="Detected service type")
    reasoning: str = Field(default="", description="Decision reasoning")


class ProactiveAllocationResult(BaseModel):
    """Complete proactive allocation result schema"""
    # Input summary
    user_id: str = Field(default="", description="User identifier")
    trajectory_length: int = Field(default=0, description="Number of trajectory points")
    
    # Prediction
    current_position: Tuple[float, float] = Field(description="Current (x, y) position")
    predicted_position: Tuple[float, float] = Field(description="Predicted (x, y) position")
    predicted_velocity: Tuple[float, float] = Field(description="Estimated (vx, vy) velocity")
    
    # Channel quality
    current_cqi: int = Field(default=10, description="CQI at current position")
    predicted_cqi: int = Field(default=10, description="CQI at predicted position")
    
    # Allocation
    service_type: str = Field(default="", description="Detected service type")
    slice_type: str = Field(default="eMBB", description="Allocated slice")
    allocated_bandwidth: float = Field(default=10.0, description="Bandwidth in MHz")
    expected_rate: float = Field(default=100.0, description="Rate in Mbps")
    qos_satisfied: bool = Field(default=True, description="Whether QoS is met")


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
    
    # Structured results
    predicted_position: Tuple[float, float] = Field(default=(0.0, 0.0), description="Predicted position")
    predicted_cqi: int = Field(default=10, description="Predicted CQI")
    slice_type: str = Field(default="eMBB", description="Determined slice type")
    allocated_bandwidth: float = Field(default=10.0, description="Allocated bandwidth in MHz")
    expected_rate: float = Field(default=100.0, description="Expected rate in Mbps")
