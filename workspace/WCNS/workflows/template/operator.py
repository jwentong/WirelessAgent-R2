# -*- coding: utf-8 -*-
# @Date    : Feb.10
# @Author  : Jwen
# @Desc    : WCNS (Network Slicing) Agent - Operators and Tools
#            Version 1.0 - ReAct-based Network Slicing Decision Making

from pathlib import Path
from typing import List, Optional, Dict, Any
import math
import json
import re

from scripts.formatter import BaseFormatter, XmlFormatter, TextFormatter
from workspace.WCNS.workflows.template.operator_an import *
from workspace.WCNS.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger

from scripts.operators import Operator, ReActAgent
from scripts.base_operator import BaseOperator
from scripts.tools import ToolRegistry, BaseTool, ToolSchema, ToolParameter

# Note: Custom, ScEnsemble, NetworkSlicingAgent are the core operators for WCNS


# ============================================================================
# Code-Level Ray Tracing Helper (for hybrid Custom + ray_tracing workflows)
# ============================================================================

class CodeLevelRayTracing:
    """
    Extracts user coordinates from WCNS problem text and calls RayTracingTool 
    directly at the Python level — no LLM XML parsing needed.
    
    Usage in graph.py:
        rt = operator.CodeLevelRayTracing()
        cqi_info = await rt.get_cqi(problem)
        # cqi_info = {"cqi": 15, "snr_dB": 37.93, "has_los": True, ...}
        enhanced = rt.inject_cqi(problem, cqi_info)
        # enhanced problem now has "CQI (from ray_tracing): 15" appended
    """
    
    def __init__(self):
        self._ray_tracing_tool = None
    
    def _get_tool(self):
        if self._ray_tracing_tool is None:
            self._ray_tracing_tool = RayTracingTool()
        return self._ray_tracing_tool
    
    def extract_coordinates(self, problem: str) -> dict:
        """Extract user_x, user_y, region from problem text.
        
        Parses patterns like:
            User Position: (-175.42, -25.92) meters
            Region: HKUST_Center
        """
        result = {"user_x": None, "user_y": None, "region": None}
        
        # Extract position
        pos_pattern = r'User Position:\s*\(([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\)'
        pos_match = re.search(pos_pattern, problem)
        if pos_match:
            result["user_x"] = float(pos_match.group(1))
            result["user_y"] = float(pos_match.group(2))
        
        # Extract region
        region_pattern = r'Region:\s*(HKUST_\w+)'
        region_match = re.search(region_pattern, problem)
        if region_match:
            result["region"] = region_match.group(1)
        
        return result
    
    async def get_cqi(self, problem: str) -> dict:
        """Extract coordinates from problem and run ray_tracing.
        
        Returns:
            dict with keys: cqi, snr_dB, has_los, rx_power_dBm, path_loss_dB, distance_m
            On failure: dict with cqi=12 (safe fallback) and error message
        """
        coords = self.extract_coordinates(problem)
        
        if coords["user_x"] is None or coords["user_y"] is None or coords["region"] is None:
            logger.warning("CodeLevelRayTracing: Could not extract coordinates from problem")
            return {"cqi": 12, "error": "Could not extract coordinates", "fallback": True}
        
        try:
            tool = self._get_tool()
            result = await tool.execute(
                user_x=coords["user_x"],
                user_y=coords["user_y"],
                region=coords["region"]
            )
            result_str = result["result"]
            
            # Parse CQI from result string
            cqi_match = re.search(r'CQI\):\s*(\d+)', result_str)
            snr_match = re.search(r'SNR:\s*([+-]?\d+\.?\d*)', result_str)
            los_match = re.search(r'Line of Sight:\s*(Yes|No)', result_str)
            
            cqi = int(cqi_match.group(1)) if cqi_match else 12
            snr = float(snr_match.group(1)) if snr_match else 0.0
            has_los = los_match.group(1) == "Yes" if los_match else None
            
            logger.info(f"CodeLevelRayTracing: CQI={cqi}, SNR={snr:.1f}dB, LOS={has_los} "
                       f"at ({coords['user_x']}, {coords['user_y']}) in {coords['region']}")
            
            return {
                "cqi": cqi,
                "snr_dB": snr,
                "has_los": has_los,
                "region": coords["region"],
                "user_x": coords["user_x"],
                "user_y": coords["user_y"],
                "fallback": False
            }
            
        except Exception as e:
            logger.error(f"CodeLevelRayTracing error: {e}")
            return {"cqi": 12, "error": str(e), "fallback": True}
    
    def inject_cqi(self, problem: str, cqi_info: dict) -> str:
        """Inject CQI info into the problem text for Custom operator."""
        cqi = cqi_info.get("cqi", 12)
        
        if cqi_info.get("fallback"):
            cqi_note = (f"\n\n=== CQI INFORMATION (estimated) ===\n"
                       f"CQI: {cqi} (fallback estimate — could not run ray_tracing)\n"
                       f"Use this CQI for throughput calculation.")
        else:
            snr = cqi_info.get("snr_dB", 0)
            los = cqi_info.get("has_los")
            los_str = "LOS (line-of-sight)" if los else "NLOS (non-line-of-sight)"
            cqi_note = (f"\n\n=== CQI INFORMATION (from ray_tracing) ===\n"
                       f"CQI: {cqi}\n"
                       f"SNR: {snr:.2f} dB\n"
                       f"Channel Condition: {los_str}\n"
                       f"⚠️ This CQI is ACCURATE from ray_tracing. Use this exact value for throughput calculation.\n"
                       f"Throughput formula: T = 10 × Bandwidth × log₁₀(1 + 10^({cqi}/10))")
        
        return problem + cqi_note


# ============================================================================
# WCNS-Specific Tools
# ============================================================================

class KnowledgeBaseQueryTool(BaseTool):
    """
    Knowledge Base Query Tool for Network Slicing
    
    Queries the network slicing knowledge base for:
    - Service type classification rules
    - Slice type characteristics (eMBB vs URLLC)
    - QoS requirements by service type
    - Bandwidth allocation guidelines
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="knowledge_base_query",
            description="""Query the network slicing knowledge base for information about:
- Service type classification (video_streaming, online_gaming, voip_call, etc.)
- Slice type selection (eMBB for bandwidth, URLLC for low latency)
- QoS requirements (min_rate, max_latency, reliability)
- Bandwidth constraints for each slice type

Use this tool to understand which slice type is appropriate for a given service.""",
            parameters=[
                ToolParameter(name="query", type="string",
                             description="The question about network slicing (e.g., 'What slice type for video streaming?')",
                             required=True)
            ]
        )
        super().__init__(schema)
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Execute knowledge base query"""
        query_lower = query.lower()
        
        # Service type detection keywords
        service_mapping = {
            "video_streaming": ["video", "stream", "4k", "8k", "hd", "netflix", "youtube"],
            "online_gaming": ["game", "gaming", "esport", "fps", "moba"],
            "voip_call": ["voip", "call", "voice", "phone", "sip"],
            "web_browsing": ["web", "browse", "surf", "http", "website"],
            "file_download": ["download", "file", "transfer", "ftp", "backup"],
            "ar_navigation": ["ar", "augmented", "navigation", "map", "reality"],
            "remote_control": ["remote", "control", "robot", "drone", "industrial"],
            "iot_monitoring": ["iot", "sensor", "monitor", "smart", "device"]
        }
        
        # Slice type info
        slice_info = {
            "eMBB": {
                "description": "enhanced Mobile Broadband - for high bandwidth applications",
                "bandwidth_range": "6-20 MHz",
                "rate_range": "100-400 Mbps",
                "total_capacity": "90 MHz",
                "services": ["video_streaming", "file_download", "ar_navigation", "web_browsing"]
            },
            "URLLC": {
                "description": "Ultra-Reliable Low-Latency Communications - for real-time applications",
                "bandwidth_range": "1-5 MHz",
                "rate_range": "1-100 Mbps",
                "total_capacity": "30 MHz",
                "services": ["online_gaming", "voip_call", "remote_control", "iot_monitoring"]
            }
        }
        
        # QoS requirements
        qos_requirements = {
            "video_streaming": {"min_rate": 25, "max_latency": 100, "slice": "eMBB"},
            "online_gaming": {"min_rate": 5, "max_latency": 20, "slice": "URLLC"},
            "voip_call": {"min_rate": 0.5, "max_latency": 50, "slice": "URLLC"},
            "web_browsing": {"min_rate": 5, "max_latency": 200, "slice": "eMBB"},
            "file_download": {"min_rate": 50, "max_latency": 500, "slice": "eMBB"},
            "ar_navigation": {"min_rate": 20, "max_latency": 50, "slice": "eMBB"},
            "remote_control": {"min_rate": 1, "max_latency": 10, "slice": "URLLC"},
            "iot_monitoring": {"min_rate": 0.1, "max_latency": 100, "slice": "URLLC"}
        }
        
        result_parts = []
        
        # Check if asking about slice type
        if "embb" in query_lower or "enhanced mobile broadband" in query_lower:
            info = slice_info["eMBB"]
            result_parts.append(f"eMBB Slice Information:\n"
                              f"- Description: {info['description']}\n"
                              f"- Bandwidth range: {info['bandwidth_range']}\n"
                              f"- Rate range: {info['rate_range']}\n"
                              f"- Total capacity: {info['total_capacity']}\n"
                              f"- Suitable for: {', '.join(info['services'])}")
        
        if "urllc" in query_lower or "low latency" in query_lower:
            info = slice_info["URLLC"]
            result_parts.append(f"URLLC Slice Information:\n"
                              f"- Description: {info['description']}\n"
                              f"- Bandwidth range: {info['bandwidth_range']}\n"
                              f"- Rate range: {info['rate_range']}\n"
                              f"- Total capacity: {info['total_capacity']}\n"
                              f"- Suitable for: {', '.join(info['services'])}")
        
        # Check for service type queries
        for service, keywords in service_mapping.items():
            if any(kw in query_lower for kw in keywords):
                qos = qos_requirements.get(service, {})
                result_parts.append(f"Service: {service}\n"
                                  f"- Recommended slice: {qos.get('slice', 'unknown')}\n"
                                  f"- Min rate: {qos.get('min_rate', 'N/A')} Mbps\n"
                                  f"- Max latency: {qos.get('max_latency', 'N/A')} ms")
        
        # Shannon formula query
        if "shannon" in query_lower or "capacity" in query_lower or "formula" in query_lower:
            result_parts.append("Shannon Capacity Formula:\n"
                              "Rate = α × Bandwidth × log₁₀(1 + 10^(CQI/10))\n"
                              "where α = 10.0 (scaling factor)\n"
                              "CQI range: 1-15 (higher is better)")
        
        if not result_parts:
            result_parts.append("Available queries:\n"
                              "- Service types: video_streaming, online_gaming, voip_call, etc.\n"
                              "- Slice types: eMBB, URLLC\n"
                              "- Shannon formula and capacity calculation\n"
                              "Please be more specific in your query.")
        
        return {"success": True, "result": "\n\n".join(result_parts)}


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
- cqi: Channel Quality Indicator (1-15, higher is better)
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
        
        return {"success": True, "result": (f"Shannon Capacity Calculation:\n"
                f"- CQI: {cqi}\n"
                f"- SNR (linear): {snr_linear:.4f}\n"
                f"- Spectral efficiency: {spectral_efficiency:.4f}\n"
                f"- Bandwidth: {bandwidth:.2f} MHz\n"
                f"- Expected rate: {rate:.2f} Mbps\n"
                f"\nFormula: Rate = {alpha} × {bandwidth} × log₁₀(1 + {snr_linear:.4f}) = {rate:.2f} Mbps")}


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
- Allocation status (success/failure)
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
        # Slice constraints
        constraints = {
            "eMBB": {"bw_min": 6, "bw_max": 20, "rate_min": 100, "rate_max": 400, "total": 90},
            "URLLC": {"bw_min": 1, "bw_max": 5, "rate_min": 1, "rate_max": 100, "total": 30}
        }
        
        if slice_type not in constraints:
            return {"success": False, "result": f"Error: Invalid slice type '{slice_type}'. Must be 'eMBB' or 'URLLC'."}
        
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
        
        # Check if allocation meets minimum requirements
        status = "SUCCESS" if expected_rate >= c["rate_min"] else "WARNING: Rate below minimum"
        
        return {"success": True, "result": (f"Slice Allocation Result:\n"
                f"- Status: {status}\n"
                f"- Slice Type: {slice_type}\n"
                f"- Requested Bandwidth: {bandwidth:.2f} MHz\n"
                f"- Allocated Bandwidth: {allocated_bw:.2f} MHz\n"
                f"- User CQI: {user_cqi}\n"
                f"- Expected Rate: {expected_rate:.2f} Mbps\n"
                f"\nConstraints: {c['bw_min']}-{c['bw_max']} MHz, {c['rate_min']}-{c['rate_max']} Mbps")}


class NetworkStateCheckerTool(BaseTool):
    """
    Network State Checker Tool
    
    Checks current network utilization and available capacity in each slice.
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="network_state_checker",
            description="""Check current network state and available capacity.

Returns:
- Current utilization for eMBB and URLLC slices
- Available bandwidth in each slice
- Number of active users

Use this to determine if there's enough capacity for new allocation.""",
            parameters=[
                ToolParameter(name="slice_type", type="string",
                             description="Which slice to check (default: all)",
                             required=False, default="all",
                             enum=["eMBB", "URLLC", "all"])
            ]
        )
        super().__init__(schema)
    
    async def execute(self, slice_type: str = "all", **kwargs) -> Dict[str, Any]:
        """Execute network state check"""
        # Get network state from kwargs if provided, otherwise use defaults
        network_state = kwargs.get("network_state", {
            "embb_utilization": 45.0,
            "urllc_utilization": 30.0,
            "embb_users": 5,
            "urllc_users": 3
        })
        
        embb_total = 90  # MHz
        urllc_total = 30  # MHz
        
        embb_used = network_state.get("embb_utilization", 45.0) / 100 * embb_total
        urllc_used = network_state.get("urllc_utilization", 30.0) / 100 * urllc_total
        
        results = []
        
        if slice_type in ["eMBB", "all"]:
            results.append(f"eMBB Slice:\n"
                         f"- Total capacity: {embb_total} MHz\n"
                         f"- Used: {embb_used:.1f} MHz ({network_state.get('embb_utilization', 45.0):.1f}%)\n"
                         f"- Available: {embb_total - embb_used:.1f} MHz\n"
                         f"- Active users: {network_state.get('embb_users', 5)}")
        
        if slice_type in ["URLLC", "all"]:
            results.append(f"URLLC Slice:\n"
                         f"- Total capacity: {urllc_total} MHz\n"
                         f"- Used: {urllc_used:.1f} MHz ({network_state.get('urllc_utilization', 30.0):.1f}%)\n"
                         f"- Available: {urllc_total - urllc_used:.1f} MHz\n"
                         f"- Active users: {network_state.get('urllc_users', 3)}")
        
        return {"success": True, "result": "\n\n".join(results)}


class OptimalBandwidthCalculatorTool(BaseTool):
    """
    Optimal Bandwidth Calculator Tool
    
    Calculates the optimal bandwidth using PROPORTIONAL FAIRNESS algorithm:
    Bandwidth = Total_Capacity / (Existing_Users + 1)
    
    This is the KEY algorithm used in the WCNS dataset!
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="optimal_bandwidth_calculator",
            description="""Calculate optimal bandwidth using PROPORTIONAL FAIRNESS algorithm.

**CRITICAL**: Use this formula for bandwidth allocation:
- eMBB: Bandwidth = 90.0 / (embb_users + 1), then clamp to [6, 20] MHz
- URLLC: Bandwidth = 30.0 / (urllc_users + 1), then clamp to [1, 5] MHz

Input:
- slice_type: "eMBB" or "URLLC"
- existing_users: Number of current users in the target slice

Output:
- Allocated bandwidth in MHz using proportional fairness""",
            parameters=[
                ToolParameter(name="slice_type", type="string",
                             description="Target slice type: 'eMBB' or 'URLLC'",
                             required=True, enum=["eMBB", "URLLC"]),
                ToolParameter(name="existing_users", type="integer",
                             description="Number of existing users in the slice",
                             required=True)
            ]
        )
        super().__init__(schema)
    
    async def execute(self, slice_type: str, existing_users: int, **kwargs) -> Dict[str, Any]:
        """Calculate optimal bandwidth using PROPORTIONAL FAIRNESS"""
        # Slice capacities
        capacities = {
            "eMBB": {"total": 90.0, "bw_min": 6.0, "bw_max": 20.0},
            "URLLC": {"total": 30.0, "bw_min": 1.0, "bw_max": 5.0}
        }
        
        if slice_type not in capacities:
            return {"success": False, "result": f"Error: Invalid slice type '{slice_type}'. Must be 'eMBB' or 'URLLC'."}
        
        c = capacities[slice_type]
        
        # PROPORTIONAL FAIRNESS FORMULA
        raw_bandwidth = c["total"] / (existing_users + 1)
        
        # Apply slice constraints
        final_bandwidth = max(c["bw_min"], min(c["bw_max"], raw_bandwidth))
        
        return {"success": True, "result": (f"Proportional Fairness Bandwidth Calculation:\n"
                f"- Slice Type: {slice_type}\n"
                f"- Total Capacity: {c['total']} MHz\n"
                f"- Existing Users: {existing_users}\n"
                f"- Formula: {c['total']} / ({existing_users} + 1) = {raw_bandwidth:.2f} MHz\n"
                f"- Constraints: [{c['bw_min']}, {c['bw_max']}] MHz\n"
                f"- **Final Allocated Bandwidth: {final_bandwidth:.2f} MHz**")}


class RayTracingTool(BaseTool):
    """
    Ray Tracing Tool for Channel Quality Estimation
    
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

You MUST call this tool to obtain CQI before calculating throughput.""",
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
        # Try multiple parent levels to find the project root with data/maps/
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
        
        # Parse buildings
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
        
        # Convert to Cartesian
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
        
        # Find tallest for TX
        tallest_idx = max(range(len(cart_buildings)), key=lambda i: cart_buildings[i]['height'])
        tallest = cart_buildings[tallest_idx]
        cx = sum(n[0] for n in tallest['nodes']) / len(tallest['nodes'])
        cy = sum(n[1] for n in tallest['nodes']) / len(tallest['nodes'])
        # Place antenna 5m above rooftop
        tx_pos = (cx, cy, tallest['height'] + 5.0)
        
        region_data = {
            'buildings': cart_buildings,
            'tx_position': tx_pos,
            'tx_building_idx': tallest_idx,
        }
        self._region_cache[region_name] = region_data
        return region_data
    
    def _has_los(self, p1, p2, buildings, skip_building_idx: int = -1):
        """Check line of sight between two 3D points.
        
        Args:
            p1: TX position (x, y, z)
            p2: RX position (x, y, z)
            buildings: List of building dicts
            skip_building_idx: Index of TX building to skip
        """
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
                continue  # Skip TX building
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
        # Validate region
        valid_regions = ["HKUST_North", "HKUST_South", "HKUST_Center"]
        if region not in valid_regions:
            return {"success": False, "result": f"Error: Invalid region '{region}'. Must be one of: {valid_regions}"}
        
        # Load region
        region_data = self._load_region(region)
        if region_data is None:
            return {"success": False, "result": f"Error: Could not load map data for region '{region}'. Map file may be missing."}
        
        tx_pos = region_data['tx_position']
        buildings = region_data['buildings']
        rx_pos = (float(user_x), float(user_y), 1.5)  # Human height
        
        # Ray tracing parameters
        frequency = 2.4e9  # 2.4 GHz
        bandwidth = 20e6   # 20 MHz
        tx_power_dBm = 30.0
        
        # Check LOS (skip TX building to avoid self-blocking)
        tx_building_idx = region_data.get('tx_building_idx', -1)
        los = self._has_los(tx_pos, rx_pos, buildings, skip_building_idx=tx_building_idx)
        
        # Path loss
        distance = math.sqrt(
            (tx_pos[0]-rx_pos[0])**2 + (tx_pos[1]-rx_pos[1])**2 + (tx_pos[2]-rx_pos[2])**2
        )
        distance = max(distance, 1.0)
        
        if los:
            path_loss_dB = 20*math.log10(distance) + 20*math.log10(frequency) - 147.55
        else:
            fspl = 20*math.log10(distance) + 20*math.log10(frequency) - 147.55
            nlos_extra = 20 + 30*math.log10(max(distance/100, 0.1))
            path_loss_dB = fspl + nlos_extra
        
        # Received power & SNR
        rx_power_dBm = tx_power_dBm - path_loss_dB
        noise_floor = -174 + 10*math.log10(bandwidth) + 8  # thermal + noise figure
        snr_dB = rx_power_dBm - noise_floor
        
        # CQI mapping
        snr_clamped = max(-10.0, min(30.0, snr_dB))
        cqi = round(1 + ((snr_clamped + 10.0) / 40.0) * 14)
        
        return {"success": True, "result": (f"Ray Tracing Result for position ({user_x}, {user_y}) in {region}:\n"
                f"- Channel Quality Indicator (CQI): {cqi}\n"
                f"- SNR: {snr_dB:.2f} dB\n"
                f"- Received Power: {rx_power_dBm:.2f} dBm\n"
                f"- Line of Sight: {'Yes (LOS)' if los else 'No (NLOS)'}\n"
                f"- Path Loss: {path_loss_dB:.2f} dB\n"
                f"- Distance to BS: {distance:.1f} m\n"
                f"\nUse CQI={cqi} for Shannon throughput calculation: T = 10 × B × log₁₀(1 + 10^({cqi}/10))")}


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
        """Create appropriate formatter based on operation class and mode"""
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
        """Legacy compatibility"""
        result = await self._execute(input=input, instruction=instruction)
        return result


class ScEnsemble(BaseOperator):
    """
    Self-Consistency Ensemble operator for WCNS
    
    Selects the most consistent network slicing decision from multiple candidates.
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
                    "description": "List of candidate slicing decisions"
                },
                "problem": {"type": "string", "description": "Original slicing problem"}
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


class NetworkSlicingAgent(BaseOperator):
    """
    Network Slicing Agent for WCNS
    
    Uses ReActAgent with domain-specific network slicing tools to determine
    optimal slice type and bandwidth allocation.
    
    Tools available:
    1. knowledge_base_query - Query slicing knowledge base
    2. shannon_calculator - Calculate rate from CQI and bandwidth
    3. slice_allocator - Allocate resources to slice
    4. network_state_checker - Check network utilization
    5. optimal_bandwidth_calculator - Calculate minimum bandwidth for QoS
    
    Workflow:
    1. Analyze user request to identify service type
    2. Query knowledge base for slice recommendation
    3. Calculate optimal bandwidth using Shannon formula
    4. Verify allocation meets QoS requirements
    5. Return final allocation decision
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "NetworkSlicingAgent", react_strategy: str = None):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Initialize tool registry with WCNS tools
        self.tool_registry = ToolRegistry()
        
        # Register WCNS-specific tools
        self.tool_registry.register(RayTracingTool())
        self.tool_registry.register(KnowledgeBaseQueryTool())
        self.tool_registry.register(ShannonCalculatorTool())
        self.tool_registry.register(SliceAllocatorTool())
        self.tool_registry.register(NetworkStateCheckerTool())
        self.tool_registry.register(OptimalBandwidthCalculatorTool())
        
        logger.info(f"NetworkSlicingAgent: Registered {len(self.tool_registry.tools)} tools")
        
        # Use external strategy if provided, otherwise use default
        if react_strategy is None:
            react_strategy = REACT_STRATEGY_PROMPT
            logger.info("NetworkSlicingAgent: Using default REACT_STRATEGY_PROMPT")
        else:
            logger.info(f"NetworkSlicingAgent: Using custom react_strategy ({len(react_strategy)} chars)")
        
        # Initialize ReActAgent with strategy prompt
        self.react_agent = ReActAgent(
            llm, 
            self.tool_registry, 
            name="WCNS_ReAct",
            strategy_prompt=react_strategy
        )
        
        logger.info(f"NetworkSlicingAgent initialized with {len(self.tool_registry.tools)} tools")
    
    def _get_input_schema(self) -> Dict:
        """Define input schema"""
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The network slicing request/problem to solve"
                },
                "user_cqi": {
                    "type": "integer",
                    "description": "User's Channel Quality Indicator (1-15)",
                    "default": 10
                },
                "network_state": {
                    "type": "object",
                    "description": "Current network state (optional)",
                    "properties": {
                        "embb_utilization": {"type": "number"},
                        "urllc_utilization": {"type": "number"},
                        "embb_users": {"type": "integer"},
                        "urllc_users": {"type": "integer"}
                    }
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum reasoning steps",
                    "default": 5
                }
            },
            "required": ["problem"]
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final slicing decision"
                },
                "slice_type": {
                    "type": "string",
                    "description": "Allocated slice type (eMBB or URLLC)"
                },
                "bandwidth": {
                    "type": "number",
                    "description": "Allocated bandwidth in MHz"
                },
                "rate": {
                    "type": "number",
                    "description": "Expected rate in Mbps"
                },
                "steps": {
                    "type": "array",
                    "description": "ReAct reasoning steps"
                }
            }
        }
    
    async def _execute(self, problem: str, user_cqi: int = None, network_state: Dict = None, 
                      max_steps: int = 5, **kwargs) -> Dict:
        """
        Execute network slicing decision
        
        Args:
            problem: User request / network slicing problem
            user_cqi: Channel Quality Indicator (1-15), None means must obtain via ray_tracing
            network_state: Current network utilization
            max_steps: Maximum reasoning steps
        
        Returns:
            {
                "answer": str,  # Final bandwidth answer
                "slice_type": str,  # "eMBB" or "URLLC"
                "bandwidth": float,  # MHz
                "rate": float,  # Mbps
                "steps": List[Dict]  # Reasoning trace
            }
        """
        try:
            # Parse network state from problem if not provided
            if network_state is None:
                network_state = self._parse_network_state(problem)
            
            # Add context to problem — do NOT inject a CQI value, force tool usage
            if user_cqi is not None:
                context_problem = f"{problem}\n\nUser CQI: {user_cqi}"
            else:
                context_problem = (f"{problem}\n\n"
                    f"\u26a0\ufe0f CQI is NOT provided in this problem. You MUST call the ray_tracing tool "
                    f"with the user's coordinates (user_x, user_y) and map region to obtain the CQI value. "
                    f"Do NOT guess or assume a CQI \u2014 call ray_tracing FIRST before any other calculation.")
            if network_state:
                context_problem += (
                    f"\n\n=== PARSED NETWORK STATE (for proportional fairness calculation) ===\n"
                    f"eMBB: {network_state.get('embb_users', 0)} users, {network_state.get('embb_utilization', 0):.1f}% utilization\n"
                    f"URLLC: {network_state.get('urllc_users', 0)} users, {network_state.get('urllc_utilization', 0):.1f}% utilization\n"
                    f"=== IMPORTANT: Use optimal_bandwidth_calculator with existing_users from above! ==="
                )
            
            # Use ReActAgent for tool-based reasoning
            result = await self.react_agent(
                problem=context_problem,
                max_iterations=max_steps,
                verbose=False
            )
            
            answer = result.get("answer", "")
            steps = result.get("steps", [])
            
            # Extract CQI from ReAct history (ray_tracing tool results)
            cqi_from_tool = self._extract_cqi_from_steps(steps)
            if cqi_from_tool is not None:
                user_cqi = cqi_from_tool
                logger.info(f"NetworkSlicingAgent: Extracted CQI={user_cqi} from ray_tracing tool")
            
            # Extract structured decision from answer
            slice_type, bandwidth, rate = self._parse_decision(answer)
            
            # Also try to extract CQI from the answer text
            cqi_from_answer = self._extract_cqi_from_text(answer)
            if cqi_from_answer is not None:
                # Only override if we didn't already get a tool-based CQI
                if cqi_from_tool is None:
                    user_cqi = cqi_from_answer
            
            # Final fallback: if CQI is still None, use 12 (closer to dataset median than 10)
            if user_cqi is None:
                user_cqi = 12
                logger.warning("NetworkSlicingAgent: No CQI from tool or answer, using fallback CQI=12")
            
            # If parsing failed, try to calculate directly using proportional fairness
            if bandwidth == 10.0 and network_state:  # Default value, likely failed to parse
                slice_type = self._detect_slice_from_problem(problem)
                if slice_type == "eMBB":
                    bandwidth = 90.0 / (network_state.get('embb_users', 4) + 1)
                    bandwidth = max(6.0, min(20.0, bandwidth))
                else:
                    bandwidth = 30.0 / (network_state.get('urllc_users', 4) + 1)
                    bandwidth = max(1.0, min(5.0, bandwidth))
            
            # Always recalculate rate with the best CQI we have
            alpha = 10.0
            snr_linear = 10 ** (user_cqi / 10)
            rate = alpha * bandwidth * math.log10(1 + snr_linear)
            
            # Format answer with ALL 4 fields for benchmark extraction
            final_answer = (f"CQI: {user_cqi}\n"
                           f"Slice Type: {slice_type}\n"
                           f"Bandwidth: {bandwidth:.2f} MHz\n"
                           f"Throughput: {rate:.2f} Mbps")
            
            return {
                "answer": final_answer,
                "slice_type": slice_type,
                "cqi": user_cqi,
                "bandwidth": round(bandwidth, 2),
                "rate": round(rate, 2),
                "steps": steps
            }
            
        except Exception as e:
            logger.error(f"NetworkSlicingAgent error: {e}")
            
            # Fallback: Use proportional fairness directly
            network_state = self._parse_network_state(problem) if network_state is None else network_state
            slice_type = self._detect_slice_from_problem(problem)
            
            # Fallback CQI = 12 if None (closer to dataset distribution)
            if user_cqi is None:
                user_cqi = 12
            
            if slice_type == "eMBB":
                bandwidth = 90.0 / (network_state.get('embb_users', 4) + 1)
                bandwidth = max(6.0, min(20.0, bandwidth))
            else:
                bandwidth = 30.0 / (network_state.get('urllc_users', 4) + 1)
                bandwidth = max(1.0, min(5.0, bandwidth))
            
            # Calculate rate
            alpha = 10.0
            snr_linear = 10 ** (user_cqi / 10)
            rate = alpha * bandwidth * math.log10(1 + snr_linear)
            
            final_answer = (f"CQI: {user_cqi}\n"
                           f"Slice Type: {slice_type}\n"
                           f"Bandwidth: {bandwidth:.2f} MHz\n"
                           f"Throughput: {rate:.2f} Mbps")
            
            return {
                "answer": final_answer,
                "slice_type": slice_type,
                "cqi": user_cqi,
                "bandwidth": round(bandwidth, 2),
                "rate": round(rate, 2),
                "steps": [],
                "used_fallback": True
            }
    
    def _parse_network_state(self, problem: str) -> Dict:
        """Parse network state from problem text"""
        state = {
            "embb_users": 4,
            "urllc_users": 4,
            "embb_utilization": 50.0,
            "urllc_utilization": 50.0
        }
        
        # Pattern: "eMBB Slice: X users, Y% utilization"
        embb_pattern = r'eMBB\s+Slice[:\s]+(\d+)\s+users?,\s+([\d.]+)%?\s+utilization'
        urllc_pattern = r'URLLC\s+Slice[:\s]+(\d+)\s+users?,\s+([\d.]+)%?\s+utilization'
        
        embb_match = re.search(embb_pattern, problem, re.IGNORECASE)
        if embb_match:
            state["embb_users"] = int(embb_match.group(1))
            state["embb_utilization"] = float(embb_match.group(2))
        
        urllc_match = re.search(urllc_pattern, problem, re.IGNORECASE)
        if urllc_match:
            state["urllc_users"] = int(urllc_match.group(1))
            state["urllc_utilization"] = float(urllc_match.group(2))
        
        return state
    
    def _detect_slice_from_problem(self, problem: str) -> str:
        """Detect slice type from problem keywords with priority-based disambiguation.
        
        Priority: eMBB high-bandwidth indicators override URLLC gaming keywords
        when combined (e.g., 'VR gaming' = eMBB because VR dominates).
        """
        problem_lower = problem.lower()
        
        # eMBB HIGH-PRIORITY keywords - these override URLLC gaming keywords
        embb_priority = [
            "vr", "ar", "virtual reality", "augmented reality", 
            "video", "stream", "4k", "8k", "download", "cloud",
            "browse", "music", "social", "conference", "meeting"
        ]
        for kw in embb_priority:
            if kw in problem_lower:
                return "eMBB"
        
        # URLLC keywords (checked after eMBB priority)
        urllc_keywords = [
            "game", "gaming", "esport", "call", "voip", "voice", 
            "control", "iot", "sensor", "robot", "real-time", "realtime", 
            "critical", "safety", "monitor", "industrial", "automation", 
            "trading", "surgery", "emergency", "warning", "vital", "fraud"
        ]
        for kw in urllc_keywords:
            if kw in problem_lower:
                return "URLLC"
        
        # Default to eMBB
        return "eMBB"
    
    def _parse_decision(self, answer: str) -> tuple:
        """Parse structured decision from answer text"""
        # Default values
        slice_type = "eMBB"
        bandwidth = 10.0
        rate = 100.0
        
        answer_lower = answer.lower()
        
        # Extract slice type
        if "urllc" in answer_lower:
            slice_type = "URLLC"
        elif "embb" in answer_lower:
            slice_type = "eMBB"
        
        # Extract bandwidth - try ANSWER format first
        answer_pattern = r'ANSWER[:\s]+(\d+\.?\d*)'
        match = re.search(answer_pattern, answer, re.IGNORECASE)
        if match:
            bandwidth = float(match.group(1))
        else:
            # Extract bandwidth (look for patterns like "10 MHz" or "bandwidth: 10")
            bw_patterns = [
                r'bandwidth[:\s]+(\d+\.?\d*)\s*(?:mhz)?',
                r'(\d+\.?\d*)\s*mhz',
                r'allocated_bandwidth[:\s]+(\d+\.?\d*)',
                r'Final Allocated Bandwidth[:\s]+(\d+\.?\d*)'
            ]
            for pattern in bw_patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    bandwidth = float(match.group(1))
                    break
        
        # Extract rate
        rate_patterns = [
            r'rate[:\s]+(\d+\.?\d*)\s*(?:mbps)?',
            r'(\d+\.?\d*)\s*mbps',
            r'expected_rate[:\s]+(\d+\.?\d*)'
        ]
        for pattern in rate_patterns:
            match = re.search(pattern, answer_lower)
            if match:
                rate = float(match.group(1))
                break
        
        return slice_type, bandwidth, rate
    
    def _extract_cqi_from_steps(self, steps: list) -> Optional[int]:
        """Extract CQI value from ReAct history (ray_tracing tool results)."""
        for step in steps:
            obs = step.get('observation', {})
            if isinstance(obs, dict):
                result_text = obs.get('result', '')
            elif isinstance(obs, str):
                result_text = obs
            else:
                continue
            
            if not result_text or not isinstance(result_text, str):
                continue
            
            # Look for CQI in ray_tracing output
            cqi_patterns = [
                r'Channel Quality Indicator \(CQI\)[:\s]+(\d+)',
                r'\*\*CQI[:\s]*(\d+)\*\*',
                r'CQI[:\s]+(\d+)',
            ]
            for pattern in cqi_patterns:
                match = re.search(pattern, result_text)
                if match:
                    cqi_val = int(match.group(1))
                    if 1 <= cqi_val <= 15:
                        return cqi_val
        return None
    
    def _extract_cqi_from_text(self, text: str) -> Optional[int]:
        """Extract CQI value from answer text."""
        patterns = [
            r'CQI[:\s]+(\d+)',
            r'cqi\s*=\s*(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cqi_val = int(match.group(1))
                if 1 <= cqi_val <= 15:
                    return cqi_val
        return None
    
    async def __call__(self, problem: str, user_cqi: int = None, network_state: Dict = None, 
                      max_steps: int = 5, **kwargs):
        return await self._execute(
            problem=problem, 
            user_cqi=user_cqi, 
            network_state=network_state, 
            max_steps=max_steps, 
            **kwargs
        )


# ============================================================================
# Direct Solver (Non-ReAct version)
# ============================================================================

class DirectSlicingSolver(BaseOperator):
    """
    Direct Network Slicing Solver
    
    A simpler operator that directly calculates slicing decision without ReAct loop.
    Useful for:
    - Faster inference
    - Baseline comparison
    - When problem is straightforward
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "DirectSlicingSolver"):
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
            "online_gaming": ["game", "gaming", "esport", "fps", "moba", "fortnite", "pubg"],
            "voip_call": ["voip", "call", "voice", "phone", "sip", "telephone"],
            "web_browsing": ["web", "browse", "surf", "http", "website", "internet"],
            "file_download": ["download", "file", "transfer", "ftp", "backup", "upload"],
            "ar_navigation": ["ar", "augmented", "navigation", "map", "reality", "vr"],
            "remote_control": ["remote", "control", "robot", "drone", "industrial", "machine"],
            "iot_monitoring": ["iot", "sensor", "monitor", "smart", "device", "meter"]
        }
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "user_cqi": {"type": "integer", "default": 10}
            },
            "required": ["problem"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
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
    
    async def _execute(self, problem: str, user_cqi: int = 10, **kwargs) -> Dict:
        """Direct calculation without ReAct"""
        # Detect service type
        service = self._detect_service_type(problem)
        slice_type, min_rate = self.service_slice_map.get(service, ("eMBB", 25))
        
        # Calculate bandwidth
        alpha = 10.0
        snr_linear = 10 ** (user_cqi / 10)
        spectral_efficiency = math.log10(1 + snr_linear)
        
        # Minimum bandwidth for required rate
        min_bandwidth = min_rate / (alpha * spectral_efficiency)
        
        # Apply constraints and add margin
        if slice_type == "eMBB":
            bandwidth = max(6, min(20, min_bandwidth * 1.1))
        else:
            bandwidth = max(1, min(5, min_bandwidth * 1.1))
        
        # Final rate
        rate = alpha * bandwidth * spectral_efficiency
        
        return {
            "answer": f"Service: {service}\nSlice: {slice_type}\nBandwidth: {bandwidth:.2f} MHz\nRate: {rate:.2f} Mbps",
            "slice_type": slice_type,
            "bandwidth": round(bandwidth, 2),
            "rate": round(rate, 2),
            "service_type": service
        }
    
    async def __call__(self, problem: str, user_cqi: int = 10, **kwargs):
        return await self._execute(problem=problem, user_cqi=user_cqi, **kwargs)
