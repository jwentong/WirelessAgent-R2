# -*- coding: utf-8 -*-
# @Date    : Feb.10
# @Author  : Jwen
# @Desc    : WCMSA (Mobile Service Assurance) Agent - Operators and Tools
#            Version 1.0 - Kalman Filter based Proactive Resource Allocation

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import math
import json
import re
import numpy as np

from scripts.formatter import BaseFormatter, XmlFormatter, TextFormatter
from workspace.WCMSA.workflows.template.operator_an import *
from workspace.WCMSA.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger

from scripts.operators import Operator, ReActAgent
from scripts.base_operator import BaseOperator
from scripts.tools import ToolRegistry, BaseTool, ToolSchema, ToolParameter


# ============================================================================
# Code-Level Kalman Predictor (predicts position only — like WCNS CodeLevelRayTracing)
# ============================================================================

class CodeLevelKalmanPredictor:
    """
    Extracts trajectory from WCMSA problem text and predicts the next
    user position using a constant-velocity Kalman Filter.

    This is analogous to WCNS's CodeLevelRayTracing: it performs ONE
    code-level computation (position prediction) and injects the result
    into the problem text.  The LLM must still determine slice type,
    bandwidth, throughput, and QoS.

    Usage in graph.py:
        kalman = operator.CodeLevelKalmanPredictor()
        pos_info = kalman.predict(problem)
        enhanced = kalman.inject_prediction(problem, pos_info)
    """

    def extract_trajectory(self, problem: str) -> List[List[float]]:
        """Extract trajectory points from problem text."""
        points = []
        hist_match = re.search(
            r'Historical Positions?:\s*([\s\S]*?)(?:\n[A-Z]|\n===|\Z)', problem)
        if hist_match:
            coords = re.findall(
                r'\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)', hist_match.group(1))
            for x_s, y_s in coords:
                points.append([float(x_s), float(y_s)])

        cur_match = re.search(
            r'Current Position:\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)', problem)
        if cur_match:
            cx, cy = float(cur_match.group(1)), float(cur_match.group(2))
            # Always append — even if duplicate — to keep correct last-2 pair
            points.append([cx, cy])

        return points

    def extract_region(self, problem: str) -> Optional[str]:
        m = re.search(r'Region:\s*(HKUST_\w+)', problem)
        return m.group(1) if m else None

    def predict(self, problem: str) -> dict:
        """Predict next position using constant-velocity model.

        Returns dict with: pred_x, pred_y, region, trajectory_len
        """
        traj = self.extract_trajectory(problem)
        region = self.extract_region(problem)

        if len(traj) < 2:
            p = traj[-1] if traj else [0.0, 0.0]
            logger.warning("CodeLevelKalmanPredictor: <2 points, using last point")
            return {"pred_x": p[0], "pred_y": p[1], "region": region,
                    "trajectory_len": len(traj)}

        curr = np.array(traj[-1])
        prev = np.array(traj[-2])
        velocity = (curr - prev)
        pred = curr + velocity

        return {
            "pred_x": round(float(pred[0]), 2),
            "pred_y": round(float(pred[1]), 2),
            "region": region,
            "trajectory_len": len(traj),
        }

    def inject_prediction(self, problem: str, pos_info: dict) -> str:
        """Inject predicted position into problem text."""
        px, py = pos_info["pred_x"], pos_info["pred_y"]
        block = (
            f"\n\n=== PREDICTED POSITION (Kalman Filter) ===\n"
            f"Predicted Position: ({px:.2f}, {py:.2f})\n"
            f"Trajectory Length: {pos_info['trajectory_len']} points\n"
            f"⚠️ Use this PREDICTED position for the remaining analysis."
        )
        return problem + block


# ============================================================================
# Code-Level Ray Tracing Helper (gets CQI only — same pattern as WCNS)
# ============================================================================

class CodeLevelRayTracing:
    """
    Calls RayTracingTool directly at the Python level — no LLM XML parsing
    needed.  Returns accurate CQI at any given position.

    For WCMSA the typical flow is:
        1. CodeLevelKalmanPredictor → predicted position
        2. CodeLevelRayTracing.get_cqi_at(pred_x, pred_y, region) → CQI
        3. inject_cqi() appends CQI info to problem text
        4. Custom LLM determines slice, bandwidth, throughput, QoS

    Usage in graph.py:
        rt = operator.CodeLevelRayTracing()
        cqi_info = await rt.get_cqi_at(pos_info['pred_x'], pos_info['pred_y'],
                                        pos_info['region'])
        enhanced = rt.inject_cqi(enhanced, cqi_info)
    """

    def __init__(self):
        self._ray_tracing_tool = None

    def _get_tool(self):
        if self._ray_tracing_tool is None:
            self._ray_tracing_tool = RayTracingTool()
        return self._ray_tracing_tool

    def extract_coordinates(self, problem: str) -> dict:
        """Extract user_x, user_y, region from problem text."""
        result = {"user_x": None, "user_y": None, "region": None}
        pos_match = re.search(
            r'User Position:\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)', problem)
        if pos_match:
            result["user_x"] = float(pos_match.group(1))
            result["user_y"] = float(pos_match.group(2))
        region_match = re.search(r'Region:\s*(HKUST_\w+)', problem)
        if region_match:
            result["region"] = region_match.group(1)
        return result

    async def get_cqi(self, problem: str) -> dict:
        """Extract coordinates from problem and run ray_tracing."""
        coords = self.extract_coordinates(problem)
        if coords["user_x"] is None or coords["region"] is None:
            return {"cqi": 12, "error": "Could not extract coordinates",
                    "fallback": True}
        return await self.get_cqi_at(
            coords["user_x"], coords["user_y"], coords["region"])

    async def get_cqi_at(self, x: float, y: float, region: str) -> dict:
        """Run ray_tracing at explicit coordinates.

        Returns dict with: cqi, snr_dB, has_los, region, fallback
        """
        if region is None:
            return {"cqi": 12, "error": "No region", "fallback": True}
        try:
            tool = self._get_tool()
            result = await tool.execute(user_x=x, user_y=y, region=region)
            result_str = result["result"] if result.get("success") else ""

            cqi_match = re.search(r'CQI.*?:\s*(\d+)', result_str)
            snr_match = re.search(r'SNR:\s*([+-]?\d+\.?\d*)', result_str)
            los_match = re.search(r'Line of Sight:\s*(Yes|No)', result_str)

            cqi = int(cqi_match.group(1)) if cqi_match else 12
            snr = float(snr_match.group(1)) if snr_match else 0.0
            has_los = los_match.group(1) == "Yes" if los_match else None

            logger.info(
                f"CodeLevelRayTracing: CQI={cqi}, SNR={snr:.1f}dB, "
                f"LOS={has_los} at ({x:.2f}, {y:.2f}) in {region}")

            return {"cqi": cqi, "snr_dB": snr, "has_los": has_los,
                    "region": region, "user_x": x, "user_y": y,
                    "fallback": False}
        except Exception as e:
            logger.error(f"CodeLevelRayTracing error: {e}")
            return {"cqi": 12, "error": str(e), "fallback": True}

    def inject_cqi(self, problem: str, cqi_info: dict) -> str:
        """Inject CQI info into the problem text."""
        cqi = cqi_info.get("cqi", 12)

        if cqi_info.get("fallback"):
            cqi_note = (
                f"\n\n=== CQI INFORMATION (estimated) ===\n"
                f"Predicted CQI: {cqi} (fallback estimate)\n"
                f"Use this CQI for throughput calculation.")
        else:
            snr = cqi_info.get("snr_dB", 0)
            los = cqi_info.get("has_los")
            los_str = "LOS (line-of-sight)" if los else "NLOS (non-line-of-sight)"
            cqi_note = (
                f"\n\n=== CQI INFORMATION (from ray_tracing) ===\n"
                f"Predicted CQI: {cqi}\n"
                f"SNR: {snr:.2f} dB\n"
                f"Channel Condition: {los_str}\n"
                f"⚠️ This CQI is ACCURATE from ray_tracing. "
                f"Use this exact value for throughput calculation.\n"
                f"Throughput formula: T = 10 × Bandwidth × "
                f"log₁₀(1 + 10^({cqi}/10))")

        return problem + cqi_note


# ============================================================================
# WCMSA-Specific Tools
# ============================================================================

class KalmanFilterPredictorTool(BaseTool):
    """
    Kalman Filter Predictor Tool for Trajectory Prediction
    
    Predicts next user position based on trajectory history using
    a simple 2D constant velocity Kalman Filter.
    
    State: [x, y, vx, vy]
    Measurement: [x, y]
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="kalman_filter_predictor",
            description="""Predict next user position using Kalman Filter.

State vector: [x, y, vx, vy] (position and velocity)
Uses constant velocity model for prediction.

Input:
- trajectory_history: List of (x, y) positions, most recent last
- dt: Time step in seconds (default: 1.0)

Output:
- Predicted position (x, y)
- Estimated velocity (vx, vy)
- Prediction confidence

Use this tool to predict where the user will be in the next time step.""",
            parameters=[
                ToolParameter(name="trajectory_history", type="array",
                             description="List of (x, y) positions, most recent last",
                             required=True),
                ToolParameter(name="dt", type="number",
                             description="Time step in seconds (default: 1.0)",
                             required=False, default=1.0)
            ]
        )
        super().__init__(schema)
        # Default Kalman Filter parameters
        self.process_noise = 0.5  # Q diagonal
        self.measurement_noise = 2.0  # R diagonal
    
    async def execute(self, trajectory_history: List[List[float]], dt: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Execute Kalman Filter prediction"""
        if len(trajectory_history) < 2:
            return {"success": False, "result": None, "error": "Need at least 2 trajectory points for prediction"}
        
        try:
            # Convert to numpy array
            trajectory = np.array(trajectory_history)
            n_points = len(trajectory)
            
            # Initialize state from last two points
            pos_current = trajectory[-1]
            pos_prev = trajectory[-2]
            velocity = (pos_current - pos_prev) / dt
            
            # State: [x, y, vx, vy]
            state = np.array([pos_current[0], pos_current[1], velocity[0], velocity[1]])
            
            # State transition matrix (constant velocity model)
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Covariance matrix (initialize with some uncertainty)
            P = np.eye(4) * 10
            
            # Process noise
            Q = np.eye(4) * self.process_noise
            Q[0, 0] = Q[1, 1] = 0.1  # Lower noise for position
            Q[2, 2] = Q[3, 3] = self.process_noise  # Higher for velocity
            
            # Run Kalman Filter updates for historical points (if more than 2)
            if n_points > 2:
                H = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]
                ])
                R = np.eye(2) * self.measurement_noise
                
                for i in range(2, n_points):
                    # Prediction step
                    state = F @ state
                    P = F @ P @ F.T + Q
                    
                    # Update step with measurement
                    z = trajectory[i]
                    y = z - H @ state  # Innovation
                    S = H @ P @ H.T + R  # Innovation covariance
                    K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
                    
                    state = state + K @ y
                    P = (np.eye(4) - K @ H) @ P
            
            # Final prediction step (predict next position)
            state_pred = F @ state
            P_pred = F @ P @ F.T + Q
            
            x_pred, y_pred = state_pred[0], state_pred[1]
            vx, vy = state_pred[2], state_pred[3]
            
            # Calculate confidence (inverse of trace of position covariance)
            position_uncertainty = np.sqrt(P_pred[0, 0] + P_pred[1, 1])
            confidence = max(0.1, min(0.99, 1.0 / (1.0 + position_uncertainty / 10)))
            
            result_str = (f"Kalman Filter Prediction Results:\n"
                    f"- Trajectory length: {n_points} points\n"
                    f"- Current position: ({pos_current[0]:.2f}, {pos_current[1]:.2f})\n"
                    f"- Estimated velocity: ({vx:.2f}, {vy:.2f}) m/s\n"
                    f"- Speed: {np.sqrt(vx**2 + vy**2):.2f} m/s\n"
                    f"- **Predicted position: ({x_pred:.2f}, {y_pred:.2f})**\n"
                    f"- Position uncertainty: ±{position_uncertainty:.2f} m\n"
                    f"- Prediction confidence: {confidence:.2%}")
            return {"success": True, "result": result_str}
            
        except Exception as e:
            return {"success": False, "result": None, "error": f"Kalman Filter error: {str(e)}"}


class CQICalculatorTool(BaseTool):
    """
    CQI Calculator Tool
    
    Calculates Channel Quality Indicator (CQI) at a given position
    using a path loss model based on distance to base station.
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="cqi_calculator",
            description="""Calculate CQI (Channel Quality Indicator) at a given position.

Uses path loss model: PL(d) = PL_ref + 10 × n × log10(d)
SNR = Tx_Power - PL - Noise_Power
Maps SNR to CQI (1-15)

Input:
- position: (x, y) coordinates in meters
- bs_position: Base station position (default: (0, 0))

Output:
- CQI value (1-15)
- SNR in dB
- Path loss details""",
            parameters=[
                ToolParameter(name="position", type="array",
                             description="User position (x, y) in meters",
                             required=True),
                ToolParameter(name="bs_position", type="array",
                             description="Base station position (default: (0, 0))",
                             required=False)
            ]
        )
        super().__init__(schema)
        # Default path loss parameters
        self.pl_ref = 38.0  # Reference path loss at 1m (dB)
        self.path_loss_exponent = 3.5  # Urban environment
        self.tx_power = 46.0  # Base station transmit power (dBm)
        self.noise_power = -100.0  # Noise floor (dBm)
    
    def _snr_to_cqi(self, snr_db: float) -> int:
        """Map SNR (dB) to CQI (1-15)"""
        thresholds = [
            (-6, 1), (-4, 2), (-2, 3), (0, 4), (2, 5),
            (4, 6), (6, 7), (8, 8), (10, 9), (13, 10),
            (16, 11), (19, 12), (22, 13), (25, 14)
        ]
        
        for threshold, cqi in thresholds:
            if snr_db < threshold:
                return cqi
        return 15
    
    async def execute(self, position: List[float], bs_position: List[float] = None, **kwargs) -> Dict[str, Any]:
        """Execute CQI calculation"""
        if bs_position is None:
            bs_position = [0.0, 0.0]
        
        try:
            # Calculate distance
            dx = position[0] - bs_position[0]
            dy = position[1] - bs_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            # Minimum distance to avoid log(0)
            distance = max(distance, 1.0)
            
            # Path loss calculation
            path_loss = self.pl_ref + 10 * self.path_loss_exponent * math.log10(distance)
            
            # SNR calculation
            received_power = self.tx_power - path_loss
            snr_db = received_power - self.noise_power
            
            # Map to CQI
            cqi = self._snr_to_cqi(snr_db)
            
            result_str = (f"CQI Calculation Results:\n"
                    f"- User position: ({position[0]:.2f}, {position[1]:.2f})\n"
                    f"- Base station: ({bs_position[0]:.2f}, {bs_position[1]:.2f})\n"
                    f"- Distance: {distance:.2f} m\n"
                    f"- Path loss: {path_loss:.2f} dB\n"
                    f"- Received power: {received_power:.2f} dBm\n"
                    f"- SNR: {snr_db:.2f} dB\n"
                    f"- **CQI: {cqi}**")
            return {"success": True, "result": result_str}
            
        except Exception as e:
            return {"success": False, "result": None, "error": f"CQI calculation error: {str(e)}"}


class ShannonCalculatorTool(BaseTool):
    """
    Shannon Capacity Calculator Tool
    
    Calculates the expected rate using the Shannon formula:
    Rate = α × Bandwidth × log₁₀(1 + 10^(CQI/10))
    where α = 10.0
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="shannon_calculator",
            description="""Calculate expected throughput rate using Shannon formula.
Formula: Rate = 10.0 × Bandwidth × log₁₀(1 + 10^(CQI/10))

Input:
- cqi: Channel Quality Indicator (1-15)
- bandwidth: Allocated bandwidth in MHz

Output:
- Expected rate in Mbps

Use this to determine if allocated bandwidth meets QoS requirements.""",
            parameters=[
                ToolParameter(name="cqi", type="integer",
                             description="Channel Quality Indicator (1-15)",
                             required=True),
                ToolParameter(name="bandwidth", type="number",
                             description="Bandwidth in MHz",
                             required=True)
            ]
        )
        super().__init__(schema)
    
    async def execute(self, cqi: int, bandwidth: float, **kwargs) -> Dict[str, Any]:
        """Execute Shannon calculation"""
        alpha = 10.0
        snr_linear = 10 ** (cqi / 10)
        spectral_efficiency = math.log10(1 + snr_linear)
        rate = alpha * bandwidth * spectral_efficiency
        
        result_str = (f"Shannon Capacity Calculation:\n"
                f"- CQI: {cqi}\n"
                f"- SNR (linear): {snr_linear:.4f}\n"
                f"- Spectral efficiency: {spectral_efficiency:.4f}\n"
                f"- Bandwidth: {bandwidth:.2f} MHz\n"
                f"- **Expected rate: {rate:.2f} Mbps**\n"
                f"\nFormula: Rate = {alpha} × {bandwidth} × log₁₀(1 + {snr_linear:.4f}) = {rate:.2f} Mbps")
        return {"success": True, "result": result_str}


class SliceAllocatorTool(BaseTool):
    """
    Network Slice Allocator Tool
    
    Allocates user to a network slice with specified bandwidth.
    Validates allocation against slice constraints.
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="slice_allocator",
            description="""Allocate user to a network slice with specified bandwidth.

Slice constraints:
- eMBB: 6-20 MHz bandwidth, 100-400 Mbps rate
- URLLC: 1-5 MHz bandwidth, 1-100 Mbps rate

Input:
- slice_type: "eMBB" or "URLLC"
- bandwidth: Requested bandwidth in MHz
- user_cqi: User's CQI for rate calculation

Output:
- Allocation status
- Actual allocated bandwidth
- Expected rate""",
            parameters=[
                ToolParameter(name="slice_type", type="string",
                             description="Network slice type: 'eMBB' or 'URLLC'",
                             required=True, enum=["eMBB", "URLLC"]),
                ToolParameter(name="bandwidth", type="number",
                             description="Requested bandwidth in MHz",
                             required=True),
                ToolParameter(name="user_cqi", type="integer",
                             description="User's Channel Quality Indicator (1-15)",
                             required=True)
            ]
        )
        super().__init__(schema)
    
    async def execute(self, slice_type: str, bandwidth: float, user_cqi: int, **kwargs) -> Dict[str, Any]:
        """Execute slice allocation"""
        constraints = {
            "eMBB": {"bw_min": 6, "bw_max": 20, "rate_min": 100, "rate_max": 400, "total": 90},
            "URLLC": {"bw_min": 1, "bw_max": 5, "rate_min": 1, "rate_max": 100, "total": 30}
        }
        
        if slice_type not in constraints:
            return {"success": False, "result": None, "error": f"Invalid slice type '{slice_type}'. Must be 'eMBB' or 'URLLC'."}
        
        c = constraints[slice_type]
        
        # Validate and clamp bandwidth
        allocated_bw = max(c["bw_min"], min(c["bw_max"], bandwidth))
        
        # Calculate expected rate
        alpha = 10.0
        snr_linear = 10 ** (user_cqi / 10)
        spectral_efficiency = math.log10(1 + snr_linear)
        expected_rate = alpha * allocated_bw * spectral_efficiency
        
        # Clamp rate to slice limits
        expected_rate = max(c["rate_min"], min(c["rate_max"], expected_rate))
        
        status = "SUCCESS" if expected_rate >= c["rate_min"] else "WARNING: Rate below minimum"
        
        result_str = (f"Slice Allocation Result:\n"
                f"- Status: {status}\n"
                f"- Slice Type: {slice_type}\n"
                f"- Requested Bandwidth: {bandwidth:.2f} MHz\n"
                f"- **Allocated Bandwidth: {allocated_bw:.2f} MHz**\n"
                f"- User CQI: {user_cqi}\n"
                f"- **Expected Rate: {expected_rate:.2f} Mbps**\n"
                f"\nConstraints: {c['bw_min']}-{c['bw_max']} MHz, {c['rate_min']}-{c['rate_max']} Mbps")
        return {"success": True, "result": result_str}


class KnowledgeBaseQueryTool(BaseTool):
    """
    Knowledge Base Query Tool for Mobile Service Assurance
    
    Queries the knowledge base for:
    - Service type classification
    - QoS requirements
    - Slice characteristics
    - Kalman Filter parameters
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="knowledge_base_query",
            description="""Query the mobile service assurance knowledge base for:
- Service type classification and QoS requirements
- Slice type selection guidelines
- Kalman Filter parameters
- Shannon formula and path loss model

Use this to understand requirements for different services.""",
            parameters=[
                ToolParameter(name="query", type="string",
                             description="Question about mobile service assurance",
                             required=True)
            ]
        )
        super().__init__(schema)
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute knowledge base query"""
        query_lower = query.lower()
        
        # Service type detection
        service_qos = {
            "video_streaming": {"slice": "eMBB", "min_rate": 25, "max_latency": 100},
            "online_gaming": {"slice": "URLLC", "min_rate": 5, "max_latency": 20},
            "voip_call": {"slice": "URLLC", "min_rate": 0.5, "max_latency": 50},
            "web_browsing": {"slice": "eMBB", "min_rate": 5, "max_latency": 200},
            "file_download": {"slice": "eMBB", "min_rate": 50, "max_latency": 500},
            "ar_navigation": {"slice": "eMBB", "min_rate": 20, "max_latency": 50},
            "remote_control": {"slice": "URLLC", "min_rate": 1, "max_latency": 10},
            "iot_monitoring": {"slice": "URLLC", "min_rate": 0.1, "max_latency": 100}
        }
        
        result_parts = []
        
        # Check for service queries
        for service, qos in service_qos.items():
            keywords = service.replace("_", " ").split()
            if any(kw in query_lower for kw in keywords):
                result_parts.append(f"Service: {service}\n"
                                  f"- Recommended slice: {qos['slice']}\n"
                                  f"- Min rate: {qos['min_rate']} Mbps\n"
                                  f"- Max latency: {qos['max_latency']} ms")
        
        # Kalman Filter query
        if "kalman" in query_lower or "prediction" in query_lower or "trajectory" in query_lower:
            result_parts.append("Kalman Filter Parameters:\n"
                              "- State: [x, y, vx, vy]\n"
                              "- Process noise (Q): 0.5 for velocity\n"
                              "- Measurement noise (R): 2.0 for position\n"
                              "- Model: Constant velocity")
        
        # CQI/Path loss query
        if "cqi" in query_lower or "path loss" in query_lower or "channel" in query_lower:
            result_parts.append("CQI Calculation:\n"
                              "- Path loss: PL = 38 + 35 × log10(d)\n"
                              "- SNR = Tx_Power - PL - Noise\n"
                              "- CQI range: 1-15 (maps from SNR)")
        
        # Shannon formula query
        if "shannon" in query_lower or "capacity" in query_lower or "rate" in query_lower:
            result_parts.append("Shannon Capacity Formula:\n"
                              "Rate = 10.0 × Bandwidth × log₁₀(1 + 10^(CQI/10))")
        
        # Slice query
        if "embb" in query_lower or "urllc" in query_lower or "slice" in query_lower:
            result_parts.append("Slice Constraints:\n"
                              "- eMBB: 6-20 MHz, 100-400 Mbps (video, download, AR)\n"
                              "- URLLC: 1-5 MHz, 1-100 Mbps (gaming, VoIP, IoT)")
        
        if not result_parts:
            result_parts.append("Available topics:\n"
                              "- Service types and QoS requirements\n"
                              "- Kalman Filter trajectory prediction\n"
                              "- CQI calculation from position\n"
                              "- Shannon capacity formula\n"
                              "- Slice constraints (eMBB, URLLC)")
        
        return {"success": True, "result": "\n\n".join(result_parts)}


class RayTracingTool(BaseTool):
    """
    Ray Tracing Tool for Channel Quality Estimation (WCMSA V4)
    
    Performs ray tracing simulation using OSM building data to determine
    the Channel Quality Indicator (CQI) at a given user position.
    
    The tool:
    1. Loads the building map for the specified region
    2. Places a base station (TX) on the tallest building
    3. Checks line-of-sight between TX and user position
    4. Calculates path loss, received power, SNR, and CQI
    
    Supported regions: HKUST_North, HKUST_South, HKUST_Center
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="ray_tracing",
            description="""Perform ray tracing to obtain the Channel Quality Indicator (CQI) for a user at a given position.

This tool simulates wireless signal propagation using building data from OpenStreetMap.
It determines whether line-of-sight (LOS) exists between the base station and the user,
then calculates path loss, SNR, and maps to CQI (1-15).

Input:
- user_x: User's X coordinate in meters (local Cartesian)
- user_y: User's Y coordinate in meters (local Cartesian)
- region: Map region name ("HKUST_North", "HKUST_South", or "HKUST_Center")

Output:
- CQI value (1-15, higher is better channel quality)
- SNR in dB
- Whether line-of-sight exists
- Received power in dBm

You MUST call this tool to obtain CQI before calculating throughput.
Use the PREDICTED position (from Kalman Filter) as the user coordinates.""",
            parameters=[
                ToolParameter(name="user_x", type="number",
                             description="User's X coordinate in meters (local Cartesian)",
                             required=True),
                ToolParameter(name="user_y", type="number",
                             description="User's Y coordinate in meters (local Cartesian)",
                             required=True),
                ToolParameter(name="region", type="string",
                             description="Map region name",
                             required=True, enum=["HKUST_North", "HKUST_South", "HKUST_Center"])
            ]
        )
        super().__init__(schema)
        self._region_cache = {}  # Cache loaded regions
    
    def _load_region(self, region_name: str) -> dict:
        """Load and cache a map region."""
        if region_name in self._region_cache:
            return self._region_cache[region_name]
        
        import xml.etree.ElementTree as ET
        
        # Locate the .osm file — search up from this file to find data/maps/
        this_file = Path(__file__).resolve()
        maps_dir = None
        for i in range(3, 7):
            candidate = this_file.parents[i] / "data" / "maps"
            if candidate.exists():
                maps_dir = candidate
                break
        if maps_dir is None:
            # Last resort: use CWD
            maps_dir = Path.cwd() / "data" / "maps"
        osm_file = maps_dir / f"{region_name}.osm"
        
        if not osm_file.exists():
            return None
        
        tree = ET.parse(str(osm_file))
        root = tree.getroot()
        
        buildings_raw = []
        nodes = {}
        for node in root.findall('./node'):
            nodes[node.get('id')] = (float(node.get('lat')), float(node.get('lon')))
        
        for way in root.findall('./way'):
            is_building = any(tag.get('k') == 'building' for tag in way.findall('./tag'))
            if not is_building:
                continue
            
            height = 10.0
            for tag in way.findall('./tag'):
                if tag.get('k') == 'height':
                    try: height = float(tag.get('v'))
                    except ValueError: pass
                elif tag.get('k') == 'building:levels':
                    try: height = float(tag.get('v')) * 3.0
                    except ValueError: pass
            
            b_nodes = []
            refs = [nd.get('ref') for nd in way.findall('./nd')]
            if refs and refs[0] != refs[-1]:
                refs.append(refs[0])
            for ref in refs:
                if ref in nodes:
                    b_nodes.append(nodes[ref])
            if len(b_nodes) >= 3:
                buildings_raw.append({'nodes': b_nodes, 'height': height})
        
        if not buildings_raw:
            return None
        
        R = 6371000.0
        ref_lat = sum(n[0] for n in buildings_raw[0]['nodes']) / len(buildings_raw[0]['nodes'])
        ref_lon = sum(n[1] for n in buildings_raw[0]['nodes']) / len(buildings_raw[0]['nodes'])
        ref_lat_rad = math.radians(ref_lat)
        
        cart_buildings = []
        for b in buildings_raw:
            cart_nodes = []
            for lat, lon in b['nodes']:
                x = R * (math.radians(lon) - math.radians(ref_lon)) * math.cos(ref_lat_rad)
                y = R * (math.radians(lat) - math.radians(ref_lat))
                cart_nodes.append((x, y))
            cart_buildings.append({'nodes': cart_nodes, 'height': b['height']})
        
        tallest_idx = max(range(len(cart_buildings)), key=lambda i: cart_buildings[i]['height'])
        tallest = cart_buildings[tallest_idx]
        cx = sum(n[0] for n in tallest['nodes']) / len(tallest['nodes'])
        cy = sum(n[1] for n in tallest['nodes']) / len(tallest['nodes'])
        tx_pos = (cx, cy, tallest['height'] + 5.0)
        
        region_data = {
            'buildings': cart_buildings,
            'tx_position': tx_pos,
            'tx_building_idx': tallest_idx,
        }
        self._region_cache[region_name] = region_data
        return region_data
    
    def _has_los(self, p1, p2, buildings, skip_building_idx: int = -1):
        """Check line of sight between two 3D points."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        dist_2d = math.sqrt(dx**2 + dy**2)
        if dist_2d < 1e-6:
            return True
        
        def cross2d(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        def seg_intersect(a1, a2, b1, b2):
            d1 = cross2d(b1, b2, a1)
            d2 = cross2d(b1, b2, a2)
            d3 = cross2d(a1, a2, b1)
            d4 = cross2d(a1, a2, b2)
            if ((d1>0 and d2<0) or (d1<0 and d2>0)) and \
               ((d3>0 and d4<0) or (d3<0 and d4>0)):
                return True
            return False
        
        for idx, building in enumerate(buildings):
            if idx == skip_building_idx:
                continue
            nodes = building['nodes']
            intersects = False
            for i in range(len(nodes) - 1):
                if seg_intersect((p1[0],p1[1]), (p2[0],p2[1]), nodes[i], nodes[i+1]):
                    intersects = True
                    break
            if intersects:
                for t in [0.3, 0.5, 0.7]:
                    z_at_t = p1[2] + t * dz
                    if z_at_t < building['height']:
                        return False
        return True
    
    async def execute(self, user_x: float, user_y: float, region: str, **kwargs) -> Dict[str, Any]:
        """Execute ray tracing for the given user position and region."""
        valid_regions = ["HKUST_North", "HKUST_South", "HKUST_Center"]
        if region not in valid_regions:
            return {"success": False, "result": None, "error": f"Invalid region '{region}'. Must be one of: {valid_regions}"}
        
        region_data = self._load_region(region)
        if region_data is None:
            return {"success": False, "result": None, "error": f"Could not load map data for region '{region}'. Map file may be missing."}
        
        tx_pos = region_data['tx_position']
        buildings = region_data['buildings']
        rx_pos = (float(user_x), float(user_y), 1.5)
        
        frequency = 2.4e9
        bandwidth = 20e6
        tx_power_dBm = 30.0
        
        tx_building_idx = region_data.get('tx_building_idx', -1)
        los = self._has_los(tx_pos, rx_pos, buildings, skip_building_idx=tx_building_idx)
        
        distance = math.sqrt(
            (tx_pos[0]-rx_pos[0])**2 + (tx_pos[1]-rx_pos[1])**2 + (tx_pos[2]-rx_pos[2])**2
        )
        distance = max(distance, 1.0)
        
        if los:
            path_loss_dB = 20*math.log10(distance) + 20*math.log10(frequency) - 147.55
        else:
            fspl = 20*math.log10(distance) + 20*math.log10(frequency) - 147.55
            nlos_loss = 20 + 30*math.log10(max(distance/100, 0.1))
            path_loss_dB = fspl + nlos_loss
        
        rx_power_dBm = tx_power_dBm - path_loss_dB
        thermal_noise_dBm = -174 + 10*math.log10(bandwidth)
        noise_figure_dB = 8
        noise_floor_dBm = thermal_noise_dBm + noise_figure_dB
        snr_dB = rx_power_dBm - noise_floor_dBm
        
        min_snr, max_snr = -10.0, 30.0
        snr_clamped = max(min_snr, min(max_snr, snr_dB))
        normalized = (snr_clamped - min_snr) / (max_snr - min_snr)
        cqi = round(1 + normalized * 14)
        
        result_str = (f"Ray Tracing Results for ({user_x:.2f}, {user_y:.2f}) in {region}:\n"
                f"- Line of Sight: {'Yes (LOS)' if los else 'No (NLOS)'}\n"
                f"- Distance to BS: {distance:.1f} m\n"
                f"- Path Loss: {path_loss_dB:.1f} dB\n"
                f"- Received Power: {rx_power_dBm:.1f} dBm\n"
                f"- SNR: {snr_dB:.1f} dB\n"
                f"- **CQI: {cqi}** (1-15 scale)")
        return {"success": True, "result": result_str}


# ============================================================================
# Core Operators
# ============================================================================

class Custom(BaseOperator):
    """Custom operator with flexible instruction-based prompting"""
    
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text"},
                "instruction": {"type": "string", "description": "Instruction for processing"}
            },
            "required": ["input", "instruction"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Generated response"}
            }
        }
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Helper method for LLM calls with formatting"""
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            response = await self.llm(prompt)
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None
    
    async def _execute(self, input: str, instruction: str, **kwargs) -> Dict:
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
    async def __call__(self, input, instruction):
        result = await self._execute(input=input, instruction=instruction)
        return result


class ScEnsemble(BaseOperator):
    """
    Self-Consistency Ensemble operator for WCMSA
    
    Selects the most consistent proactive allocation from multiple candidates.
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "solutions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate allocation decisions"
                },
                "problem": {"type": "string", "description": "Original problem"}
            },
            "required": ["solutions", "problem"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Selected best solution"}
            }
        }
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            response = await self.llm(prompt)
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None

    async def _execute(self, solutions: List[str], problem: str, **kwargs) -> Dict:
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping.get(answer, 0)]}
    
    async def __call__(self, solutions: List[str], problem: str):
        return await self._execute(solutions=solutions, problem=problem)


class MobileServiceAssuranceAgent(BaseOperator):
    """
    Mobile Service Assurance Agent for WCMSA
    
    Uses ReActAgent with Kalman Filter prediction and domain-specific tools
    for proactive resource allocation based on predicted user position.
    
    Tools available:
    1. kalman_filter_predictor - Predict next position from trajectory
    2. ray_tracing - Get CQI from predicted position using real building data (MUST CALL!)
    3. cqi_calculator - Calculate CQI using simple path-loss model (legacy, less accurate)
    4. shannon_calculator - Calculate rate from CQI and bandwidth
    5. slice_allocator - Allocate resources to slice
    6. knowledge_base_query - Query service requirements
    
    Workflow:
    1. Analyze trajectory history
    2. Predict next position using Kalman Filter
    3. Call ray_tracing at predicted position to get CQI
    4. Determine service type and slice
    5. Allocate bandwidth based on predicted CQI
    6. Verify QoS requirements
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "MobileServiceAssuranceAgent", react_strategy: str = None):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Initialize tool registry with WCMSA tools
        self.tool_registry = ToolRegistry()
        
        # Register WCMSA-specific tools
        self.tool_registry.register(KalmanFilterPredictorTool())
        self.tool_registry.register(RayTracingTool())  # V4: Ray tracing for CQI (MUST CALL)
        self.tool_registry.register(CQICalculatorTool())  # Legacy: simple path-loss CQI
        self.tool_registry.register(ShannonCalculatorTool())
        self.tool_registry.register(SliceAllocatorTool())
        self.tool_registry.register(KnowledgeBaseQueryTool())
        
        logger.info(f"MobileServiceAssuranceAgent: Registered {len(self.tool_registry.tools)} tools")
        
        # Use external strategy if provided, otherwise use default
        if react_strategy is None:
            react_strategy = REACT_STRATEGY_PROMPT
            logger.info("MobileServiceAssuranceAgent: Using default REACT_STRATEGY_PROMPT")
        else:
            logger.info(f"MobileServiceAssuranceAgent: Using custom react_strategy ({len(react_strategy)} chars)")
        
        # Initialize ReActAgent
        self.react_agent = ReActAgent(
            llm, 
            self.tool_registry, 
            name="WCMSA_ReAct",
            strategy_prompt=react_strategy
        )
        
        logger.info(f"MobileServiceAssuranceAgent initialized with {len(self.tool_registry.tools)} tools")
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "User request for mobile service"
                },
                "trajectory_history": {
                    "type": "array",
                    "items": {"type": "array"},
                    "description": "List of (x, y) historical positions"
                },
                "bs_position": {
                    "type": "array",
                    "description": "Base station position (x, y)"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum reasoning steps",
                    "default": 6
                }
            },
            "required": ["problem", "trajectory_history"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "predicted_position": {"type": "array"},
                "predicted_cqi": {"type": "integer"},
                "slice_type": {"type": "string"},
                "bandwidth": {"type": "number"},
                "rate": {"type": "number"},
                "steps": {"type": "array"}
            }
        }
    
    async def _execute(self, problem: str, trajectory_history: List[List[float]], 
                      bs_position: List[float] = None, max_steps: int = 6, **kwargs) -> Dict:
        """
        Execute proactive resource allocation
        
        Args:
            problem: User request description
            trajectory_history: List of (x, y) positions
            bs_position: Base station location (default: (0, 0))
            max_steps: Maximum reasoning steps
        
        Returns:
            {
                "answer": str,
                "predicted_position": (x, y),
                "predicted_cqi": int,
                "slice_type": str,
                "bandwidth": float,
                "rate": float,
                "steps": List[Dict]
            }
        """
        if bs_position is None:
            bs_position = [0.0, 0.0]
        
        try:
            # Build context with trajectory
            trajectory_str = ", ".join([f"({p[0]:.1f}, {p[1]:.1f})" for p in trajectory_history[-5:]])
            context_problem = (f"{problem}\n\n"
                             f"Trajectory history (last {len(trajectory_history)} points): [{trajectory_str}]\n"
                             f"Base station position: ({bs_position[0]}, {bs_position[1]})")
            
            # Use ReActAgent for tool-based reasoning
            result = await self.react_agent(
                problem=context_problem,
                max_iterations=max_steps,
                verbose=False
            )
            
            answer = result.get("answer", "")
            steps = result.get("steps", [])
            
            # Extract structured decision from answer
            pred_pos, pred_cqi, slice_type, bandwidth, rate = self._parse_decision(answer, trajectory_history)
            
            return {
                "answer": answer,
                "predicted_position": pred_pos,
                "predicted_cqi": pred_cqi,
                "slice_type": slice_type,
                "bandwidth": bandwidth,
                "rate": rate,
                "steps": steps
            }
            
        except Exception as e:
            logger.error(f"MobileServiceAssuranceAgent error: {e}")
            
            # Fallback: Direct calculation
            return await self._fallback_allocation(problem, trajectory_history, bs_position)
    
    def _parse_decision(self, answer: str, trajectory_history: List[List[float]]) -> tuple:
        """Parse structured decision from answer text"""
        # Default values from simple prediction
        if len(trajectory_history) >= 2:
            pos_curr = trajectory_history[-1]
            pos_prev = trajectory_history[-2]
            pred_x = pos_curr[0] + (pos_curr[0] - pos_prev[0])
            pred_y = pos_curr[1] + (pos_curr[1] - pos_prev[1])
        else:
            pred_x, pred_y = trajectory_history[-1] if trajectory_history else (0.0, 0.0)
        
        pred_cqi = 10
        slice_type = "eMBB"
        bandwidth = 10.0
        rate = 100.0
        
        answer_lower = answer.lower()
        
        # Extract predicted position
        pos_patterns = [
            r'predicted[_ ]position[:\s]*\(?\s*(-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)',
            r'\((-?\d+\.?\d*)[,\s]+(-?\d+\.?\d*)\)\s*\*?\*?',
        ]
        for pattern in pos_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                pred_x = float(match.group(1))
                pred_y = float(match.group(2))
                break
        
        # Extract CQI
        cqi_patterns = [r'cqi[:\s]*(\d+)', r'predicted[_ ]cqi[:\s]*(\d+)']
        for pattern in cqi_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                pred_cqi = int(match.group(1))
                break
        
        # Extract slice type
        if "urllc" in answer_lower:
            slice_type = "URLLC"
        elif "embb" in answer_lower:
            slice_type = "eMBB"
        
        # Extract bandwidth
        bw_patterns = [r'bandwidth[:\s]*(\d+\.?\d*)', r'(\d+\.?\d*)\s*mhz']
        for pattern in bw_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                bandwidth = float(match.group(1))
                break
        
        # Extract rate
        rate_patterns = [r'rate[:\s]*(\d+\.?\d*)', r'(\d+\.?\d*)\s*mbps']
        for pattern in rate_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                rate = float(match.group(1))
                break
        
        return (pred_x, pred_y), pred_cqi, slice_type, bandwidth, rate
    
    async def _fallback_allocation(self, problem: str, trajectory_history: List[List[float]], 
                                   bs_position: List[float]) -> Dict:
        """Fallback direct calculation without ReAct"""
        # Kalman prediction
        if len(trajectory_history) >= 2:
            pos_curr = np.array(trajectory_history[-1])
            pos_prev = np.array(trajectory_history[-2])
            velocity = pos_curr - pos_prev
            pred_pos = pos_curr + velocity
        else:
            pred_pos = np.array(trajectory_history[-1]) if trajectory_history else np.array([0.0, 0.0])
        
        # CQI calculation
        distance = np.sqrt((pred_pos[0] - bs_position[0])**2 + (pred_pos[1] - bs_position[1])**2)
        distance = max(distance, 1.0)
        path_loss = 38.0 + 35 * np.log10(distance)
        snr_db = 46.0 - path_loss - (-100.0)
        
        # SNR to CQI mapping
        if snr_db < -6: pred_cqi = 1
        elif snr_db < 0: pred_cqi = 4
        elif snr_db < 6: pred_cqi = 7
        elif snr_db < 13: pred_cqi = 10
        elif snr_db < 20: pred_cqi = 12
        else: pred_cqi = 15
        
        # Service detection
        slice_type = "URLLC" if any(kw in problem.lower() for kw in ["game", "call", "control", "iot"]) else "eMBB"
        
        # Bandwidth allocation
        bandwidth = 10.0 if slice_type == "eMBB" else 3.0
        
        # Rate calculation
        alpha = 10.0
        snr_linear = 10 ** (pred_cqi / 10)
        rate = alpha * bandwidth * math.log10(1 + snr_linear)
        
        return {
            "answer": f"Predicted position: ({pred_pos[0]:.2f}, {pred_pos[1]:.2f}), CQI: {pred_cqi}, "
                     f"Slice: {slice_type}, Bandwidth: {bandwidth:.2f} MHz, Rate: {rate:.2f} Mbps",
            "predicted_position": (float(pred_pos[0]), float(pred_pos[1])),
            "predicted_cqi": pred_cqi,
            "slice_type": slice_type,
            "bandwidth": round(bandwidth, 2),
            "rate": round(rate, 2),
            "steps": [],
            "used_fallback": True
        }
    
    async def __call__(self, problem: str, trajectory_history: List[List[float]], 
                      bs_position: List[float] = None, max_steps: int = 6, **kwargs):
        return await self._execute(
            problem=problem, 
            trajectory_history=trajectory_history,
            bs_position=bs_position,
            max_steps=max_steps, 
            **kwargs
        )


# ============================================================================
# Direct Solver (Non-ReAct version)
# ============================================================================

class DirectMobileSolver(BaseOperator):
    """
    Direct Mobile Service Solver
    
    A simpler operator that directly calculates proactive allocation
    without ReAct loop. Useful for:
    - Faster inference
    - Baseline comparison
    - When trajectory is straightforward
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "DirectMobileSolver"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Service type to slice mapping
        self.service_slice_map = {
            "video_streaming": ("eMBB", 25),
            "online_gaming": ("URLLC", 5),
            "voip_call": ("URLLC", 0.5),
            "web_browsing": ("eMBB", 5),
            "file_download": ("eMBB", 50),
            "ar_navigation": ("eMBB", 20),
            "remote_control": ("URLLC", 1),
            "iot_monitoring": ("URLLC", 0.1)
        }
        
        # Keywords for service detection
        self.service_keywords = {
            "video_streaming": ["video", "stream", "4k", "8k", "netflix", "youtube", "hd"],
            "online_gaming": ["game", "gaming", "esport", "fps", "moba", "fortnite"],
            "voip_call": ["voip", "call", "voice", "phone", "sip", "telephone"],
            "web_browsing": ["web", "browse", "surf", "http", "website"],
            "file_download": ["download", "file", "transfer", "ftp", "backup"],
            "ar_navigation": ["ar", "augmented", "navigation", "map", "reality"],
            "remote_control": ["remote", "control", "robot", "drone", "industrial"],
            "iot_monitoring": ["iot", "sensor", "monitor", "smart", "device"]
        }
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "trajectory_history": {"type": "array"},
                "bs_position": {"type": "array"}
            },
            "required": ["problem", "trajectory_history"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "predicted_position": {"type": "array"},
                "predicted_cqi": {"type": "integer"},
                "slice_type": {"type": "string"},
                "bandwidth": {"type": "number"},
                "rate": {"type": "number"}
            }
        }
    
    def _detect_service_type(self, problem: str) -> str:
        """Detect service type from problem text"""
        problem_lower = problem.lower()
        
        for service, keywords in self.service_keywords.items():
            if any(kw in problem_lower for kw in keywords):
                return service
        
        return "video_streaming"  # Default
    
    def _kalman_predict(self, trajectory: List[List[float]], dt: float = 1.0) -> Tuple[float, float]:
        """Simple Kalman prediction"""
        if len(trajectory) < 2:
            return tuple(trajectory[-1]) if trajectory else (0.0, 0.0)
        
        pos_curr = np.array(trajectory[-1])
        pos_prev = np.array(trajectory[-2])
        velocity = (pos_curr - pos_prev) / dt
        pred_pos = pos_curr + velocity * dt
        
        return (float(pred_pos[0]), float(pred_pos[1]))
    
    def _calculate_cqi(self, position: Tuple[float, float], bs_position: List[float]) -> int:
        """Calculate CQI from position"""
        distance = math.sqrt((position[0] - bs_position[0])**2 + (position[1] - bs_position[1])**2)
        distance = max(distance, 1.0)
        
        path_loss = 38.0 + 35 * math.log10(distance)
        snr_db = 46.0 - path_loss - (-100.0)
        
        # SNR to CQI mapping
        thresholds = [(-6, 1), (-4, 2), (-2, 3), (0, 4), (2, 5), (4, 6), (6, 7), 
                     (8, 8), (10, 9), (13, 10), (16, 11), (19, 12), (22, 13), (25, 14)]
        for threshold, cqi in thresholds:
            if snr_db < threshold:
                return cqi
        return 15
    
    async def _execute(self, problem: str, trajectory_history: List[List[float]], 
                      bs_position: List[float] = None, **kwargs) -> Dict:
        """Direct calculation without ReAct"""
        if bs_position is None:
            bs_position = [0.0, 0.0]
        
        # Predict position
        pred_pos = self._kalman_predict(trajectory_history)
        
        # Calculate CQI
        pred_cqi = self._calculate_cqi(pred_pos, bs_position)
        
        # Detect service and get slice
        service = self._detect_service_type(problem)
        slice_type, min_rate = self.service_slice_map.get(service, ("eMBB", 25))
        
        # Calculate bandwidth
        alpha = 10.0
        snr_linear = 10 ** (pred_cqi / 10)
        spectral_efficiency = math.log10(1 + snr_linear)
        
        min_bandwidth = min_rate / (alpha * spectral_efficiency)
        
        # Apply constraints
        if slice_type == "eMBB":
            bandwidth = max(6, min(20, min_bandwidth * 1.1))
        else:
            bandwidth = max(1, min(5, min_bandwidth * 1.1))
        
        # Final rate
        rate = alpha * bandwidth * spectral_efficiency
        
        return {
            "answer": (f"Service: {service}\n"
                      f"Predicted position: ({pred_pos[0]:.2f}, {pred_pos[1]:.2f})\n"
                      f"Predicted CQI: {pred_cqi}\n"
                      f"Slice: {slice_type}\n"
                      f"Bandwidth: {bandwidth:.2f} MHz\n"
                      f"Rate: {rate:.2f} Mbps"),
            "predicted_position": pred_pos,
            "predicted_cqi": pred_cqi,
            "slice_type": slice_type,
            "bandwidth": round(bandwidth, 2),
            "rate": round(rate, 2),
            "service_type": service
        }
    
    async def __call__(self, problem: str, trajectory_history: List[List[float]], 
                      bs_position: List[float] = None, **kwargs):
        return await self._execute(problem=problem, trajectory_history=trajectory_history, 
                                  bs_position=bs_position, **kwargs)
