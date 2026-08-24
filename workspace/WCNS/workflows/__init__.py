# -*- coding: utf-8 -*-
"""
WCNS Workflows Module

Contains workflow templates and configurations for network slicing tasks.
"""

from workspace.WCNS.workflows.template import (
    Custom,
    ScEnsemble,
    NetworkSlicingAgent,
    DirectSlicingSolver
)

__all__ = [
    "Custom",
    "ScEnsemble",
    "NetworkSlicingAgent", 
    "DirectSlicingSolver"
]
