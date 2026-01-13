"""
WirelessAgent Agents Module
UC Berkeley AgentX Competition Submission

Author: Jingwen
Date: 1/13/2026
"""

from .green_agent import WCHWGreenAgent, run_server as run_green_agent
from .purple_agent import WCHWPurpleAgent, run_server as run_purple_agent

__all__ = [
    "WCHWGreenAgent",
    "WCHWPurpleAgent", 
    "run_green_agent",
    "run_purple_agent"
]
