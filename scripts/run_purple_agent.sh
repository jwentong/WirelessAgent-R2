#!/bin/bash
# WirelessAgent - Run Purple Agent
# UC Berkeley AgentX Competition
# Author: Jingwen
# Date: 1/13/2026

# Colors for output
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}Starting WirelessAgent Purple Agent (Baseline)...${NC}"
python agents/purple_agent.py --port 8081 --model "${1:-qwen-turbo-latest}"
