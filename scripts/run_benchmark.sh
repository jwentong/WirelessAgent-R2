#!/bin/bash
# WirelessAgent - Run Full Benchmark
# UC Berkeley AgentX Competition
# Author: Jingwen
# Date: 1/13/2026

set -e

GREEN='\033[0;32m'
PURPLE='\033[0;35m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== WirelessAgent Benchmark Runner ===${NC}"
echo ""

# Start Green Agent in background
echo -e "${GREEN}[1/3] Starting Green Agent on port 8080...${NC}"
python agents/green_agent.py --port 8080 &
GREEN_PID=$!
sleep 3

# Start Purple Agent in background
echo -e "${PURPLE}[2/3] Starting Purple Agent on port 8081...${NC}"
python agents/purple_agent.py --port 8081 &
PURPLE_PID=$!
sleep 3

# Run benchmark
echo -e "${BLUE}[3/3] Running benchmark...${NC}"
curl -X POST http://localhost:8081/run-benchmark \
  -H "Content-Type: application/json" \
  -d '{"green_agent_url": "http://localhost:8080"}'

# Cleanup
echo ""
echo -e "${BLUE}Benchmark complete. Shutting down agents...${NC}"
kill $GREEN_PID $PURPLE_PID 2>/dev/null

echo -e "${GREEN}Done!${NC}"
