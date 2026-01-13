"""
WirelessAgent Purple Agent - A2A Compatible Competition Agent
UC Berkeley AgentX Competition Submission

This is the PURPLE AGENT (baseline) that solves WCHW benchmark problems.
It demonstrates how the benchmark should be evaluated.

Author: Jingwen
Date: 1/13/2026
"""

import json
import asyncio
import logging
import requests
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.async_llm import AsyncLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PurpleAgent")


@dataclass
class AgentConfig:
    """Purple Agent Configuration"""
    model_name: str = "qwen-turbo-latest"
    temperature: float = 0.7
    max_tokens: int = 2048
    use_cot: bool = True  # Use Chain-of-Thought


class WCHWPurpleAgent:
    """
    Purple Agent (Baseline) for WCHW Benchmark
    
    This agent:
    1. Receives problems from the green agent
    2. Uses LLM to solve wireless communication problems
    3. Returns answers for evaluation
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.llm = None
        self.green_agent_url: Optional[str] = None
        
    async def initialize(self):
        """Initialize the LLM client"""
        self.llm = AsyncLLM(
            model_name=self.config.model_name,
            temperature=self.config.temperature
        )
        logger.info(f"Initialized LLM: {self.config.model_name}")
    
    def _build_prompt(self, question: str) -> str:
        """Build the prompt for solving the problem"""
        if self.config.use_cot:
            return f"""You are an expert in wireless communications. Solve the following problem step by step.

Problem:
{question}

Instructions:
1. Identify the relevant wireless communication concepts
2. List the known values and what needs to be calculated
3. Apply the appropriate formulas
4. Show your calculations step by step
5. Provide the final answer with appropriate units

Think through this carefully and provide your solution:"""
        else:
            return f"""Solve this wireless communication problem:

{question}

Provide your answer directly:"""
    
    async def solve_problem(self, question: str) -> str:
        """Solve a single problem"""
        if self.llm is None:
            await self.initialize()
        
        prompt = self._build_prompt(question)
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                max_tokens=self.config.max_tokens
            )
            
            # Extract the answer from the response
            answer = self._extract_answer(response)
            return answer
            
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            return f"Error: {str(e)}"
    
    def _extract_answer(self, response: str) -> str:
        """Extract the final answer from LLM response"""
        # Look for common answer patterns
        lines = response.strip().split('\n')
        
        # Try to find explicit answer markers
        for line in reversed(lines):
            line_lower = line.lower().strip()
            if any(marker in line_lower for marker in ['answer:', 'final answer:', 'result:', '=']):
                # Extract the value after the marker
                for marker in ['answer:', 'final answer:', 'result:']:
                    if marker in line_lower:
                        idx = line_lower.find(marker)
                        return line[idx + len(marker):].strip()
        
        # If no explicit marker, return the last non-empty line
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        
        return response.strip()
    
    async def run_benchmark(self, green_agent_url: str) -> Dict[str, Any]:
        """Run the full benchmark against a green agent"""
        self.green_agent_url = green_agent_url
        
        # Get all tasks from green agent
        logger.info(f"Fetching tasks from {green_agent_url}/tasks")
        response = requests.get(f"{green_agent_url}/tasks")
        tasks_data = response.json()
        
        tasks = tasks_data.get("tasks", [])
        logger.info(f"Received {len(tasks)} tasks")
        
        # Solve each problem
        answers = {}
        for i, task in enumerate(tasks):
            task_id = task["task_id"]
            question = task["question"]
            
            logger.info(f"Solving task {i+1}/{len(tasks)}: {task_id}")
            answer = await self.solve_problem(question)
            answers[task_id] = answer
            
            # Submit answer immediately
            submit_response = requests.post(
                f"{green_agent_url}/submit",
                json={"task_id": task_id, "answer": answer}
            )
            result = submit_response.json()
            logger.info(f"Task {task_id}: score = {result.get('score', 'N/A')}")
        
        # Get final results
        results_response = requests.get(f"{green_agent_url}/results")
        return results_response.json()


class PurpleAgentHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for Purple Agent A2A Protocol"""
    
    purple_agent: WCHWPurpleAgent = None
    
    def _set_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
    
    def _send_json(self, data: Dict, status_code: int = 200):
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "healthy", "agent": "WirelessAgent Purple Agent"})
        
        elif self.path == "/agent-card":
            self._send_json({
                "name": "WirelessAgent Purple Agent (Baseline)",
                "description": "Baseline agent for solving WCHW benchmark problems",
                "version": "1.0.0",
                "protocol": "A2A",
                "capabilities": ["problem-solving", "wireless-communication"],
                "model": self.purple_agent.config.model_name,
                "endpoints": {
                    "solve": "/solve",
                    "run-benchmark": "/run-benchmark"
                }
            })
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == "/solve":
            # Solve a single problem
            question = data.get("question", "")
            if not question:
                self._send_json({"error": "question required"}, 400)
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                answer = loop.run_until_complete(self.purple_agent.solve_problem(question))
                self._send_json({"answer": answer})
            finally:
                loop.close()
        
        elif self.path == "/run-benchmark":
            # Run full benchmark
            green_agent_url = data.get("green_agent_url")
            if not green_agent_url:
                self._send_json({"error": "green_agent_url required"}, 400)
                return
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.purple_agent.run_benchmark(green_agent_url))
                self._send_json(results)
            finally:
                loop.close()
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.address_string(), format % args))


def run_server(host: str = "0.0.0.0", port: int = 8081, model: str = "qwen-turbo-latest"):
    """Run the Purple Agent server"""
    config = AgentConfig(model_name=model)
    purple_agent = WCHWPurpleAgent(config)
    PurpleAgentHandler.purple_agent = purple_agent
    
    server = HTTPServer((host, port), PurpleAgentHandler)
    logger.info(f"Starting WirelessAgent Purple Agent on http://{host}:{port}")
    logger.info(f"Using model: {model}")
    logger.info("A2A Endpoints:")
    logger.info("  GET  /health        - Health check")
    logger.info("  GET  /agent-card    - Agent capabilities")
    logger.info("  POST /solve         - Solve a single problem")
    logger.info("  POST /run-benchmark - Run full benchmark")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Purple Agent server")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WirelessAgent Purple Agent - A2A Competition Agent")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on")
    parser.add_argument("--model", type=str, default="qwen-turbo-latest", help="LLM model to use")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, model=args.model)
