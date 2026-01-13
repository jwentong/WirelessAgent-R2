#!/bin/bash
# WirelessAgent - Run Scripts
# UC Berkeley AgentX Competition
# Author: Jingwen
# Date: 1/13/2026

# Colors for output
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting WirelessAgent Green Agent...${NC}"
python agents/green_agent.py --port 8080
