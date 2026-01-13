"""
WirelessAgent Green Agent - Server Setup and Agent Card Configuration
UC Berkeley AgentX Competition - AgentBeats Compatible

This module sets up the A2A server and defines the agent card.

Author: Jingwen
Date: 1/13/2026
"""

import asyncio
import json
import logging
from aiohttp import web

from executor import WCHWExecutor
from agent import WCHWAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WirelessAgent")

# Agent Card Configuration
AGENT_CARD = {
    "name": "WirelessAgent",
    "description": "A green agent for evaluating LLM capabilities on wireless communication problems using the WCHW (Wireless Communication Homework) benchmark.",
    "version": "1.0.0",
    "author": "Jingwen Tong",
    "protocol": "A2A",
    "url": "https://github.com/jwentong/WirelessAgent-R2",
    "skills": [
        {
            "id": "wchw-evaluation",
            "name": "WCHW Benchmark Evaluation",
            "description": "Evaluates agents on 100 wireless communication problems covering Channel Capacity, Modulation, Coding, Signal Processing, Propagation, and Noise Analysis.",
            "inputModes": ["text"],
            "outputModes": ["text"]
        }
    ],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False
    },
    "defaultInputModes": ["text"],
    "defaultOutputModes": ["text"]
}


class WirelessAgentServer:
    """A2A-compatible Green Agent Server for WCHW Benchmark"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 9009):
        self.host = host
        self.port = port
        self.agent = WCHWAgent()
        self.executor = WCHWExecutor(self.agent)
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes for A2A protocol"""
        self.app.router.add_get("/", self._handle_agent_card)
        self.app.router.add_get("/.well-known/agent.json", self._handle_agent_card)
        self.app.router.add_post("/", self._handle_request)
        self.app.router.add_get("/health", self._handle_health)
        
        # Legacy endpoints for backward compatibility
        self.app.router.add_get("/agent-card", self._handle_agent_card)
        self.app.router.add_get("/tasks", self._handle_tasks)
        self.app.router.add_post("/submit", self._handle_submit)
        self.app.router.add_get("/results", self._handle_results)
    
    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "agent": AGENT_CARD["name"],
            "version": AGENT_CARD["version"]
        })
    
    async def _handle_agent_card(self, request: web.Request) -> web.Response:
        """Return agent card (A2A discovery)"""
        return web.json_response(AGENT_CARD)
    
    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle A2A JSON-RPC style requests"""
        try:
            body = await request.json()
            result = await self.executor.execute(body)
            return web.json_response(result)
        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"code": -32700, "message": "Parse error"}},
                status=400
            )
        except Exception as e:
            logger.error(f"Request handling error: {e}")
            return web.json_response(
                {"error": {"code": -32603, "message": str(e)}},
                status=500
            )
    
    async def _handle_tasks(self, request: web.Request) -> web.Response:
        """Get all assessment tasks (legacy endpoint)"""
        tasks = self.agent.get_all_tasks()
        return web.json_response({
            "total": len(tasks),
            "tasks": tasks
        })
    
    async def _handle_submit(self, request: web.Request) -> web.Response:
        """Submit answer for evaluation (legacy endpoint)"""
        try:
            body = await request.json()
            task_id = body.get("task_id")
            answer = body.get("answer", "")
            
            if not task_id:
                return web.json_response(
                    {"error": "task_id required"}, 
                    status=400
                )
            
            result = self.agent.evaluate_response(task_id, answer)
            return web.json_response(result)
        except Exception as e:
            return web.json_response(
                {"error": str(e)}, 
                status=500
            )
    
    async def _handle_results(self, request: web.Request) -> web.Response:
        """Get current results (legacy endpoint)"""
        summary = self.agent.get_summary()
        return web.json_response(summary)
    
    def run(self):
        """Start the server"""
        logger.info(f"Starting WirelessAgent Green Agent on http://{self.host}:{self.port}")
        logger.info(f"Agent Card: GET /")
        logger.info(f"A2A Requests: POST /")
        logger.info(f"Health Check: GET /health")
        web.run_app(self.app, host=self.host, port=self.port)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="WirelessAgent Green Agent Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=9009, help="Port to listen on")
    args = parser.parse_args()
    
    server = WirelessAgentServer(host=args.host, port=args.port)
    server.run()


if __name__ == "__main__":
    main()
