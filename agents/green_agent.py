"""
WirelessAgent Green Agent - A2A Compatible Evaluation Server
UC Berkeley AgentX Competition Submission

This is the GREEN AGENT that evaluates purple agents on the WCHW benchmark.
It implements the A2A (Agent-to-Agent) protocol for communication.

Author: Jingwen
Date: 1/13/2026
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.wchw import WCHWBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GreenAgent")


@dataclass
class A2ATask:
    """A2A Protocol Task Definition"""
    task_id: str
    task_type: str = "wchw_evaluation"
    question: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class A2AResult:
    """A2A Protocol Result Definition"""
    task_id: str
    status: str  # "success", "error", "pending"
    answer: str = ""
    score: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class WCHWGreenAgent:
    """
    Green Agent for WCHW Benchmark Evaluation
    
    This agent:
    1. Loads the WCHW test dataset
    2. Sends problems to purple agents via A2A protocol
    3. Evaluates responses and computes scores
    4. Reports final benchmark results
    """
    
    def __init__(self, dataset_path: str = "data/datasets/wchw_test_70.jsonl"):
        self.dataset_path = dataset_path
        self.benchmark = WCHWBenchmark()
        self.problems: List[Dict] = []
        self.results: Dict[str, A2AResult] = {}
        self.purple_agent_url: Optional[str] = None
        
        self._load_dataset()
        
    def _load_dataset(self):
        """Load WCHW test problems"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.problems = [json.loads(line) for line in f if line.strip()]
            logger.info(f"Loaded {len(self.problems)} problems from {self.dataset_path}")
        except FileNotFoundError:
            logger.warning(f"Dataset not found: {self.dataset_path}")
            self.problems = []
    
    def create_task(self, problem_idx: int) -> A2ATask:
        """Create an A2A task from a problem"""
        if problem_idx >= len(self.problems):
            raise IndexError(f"Problem index {problem_idx} out of range")
        
        problem = self.problems[problem_idx]
        return A2ATask(
            task_id=problem.get("id", f"task_{problem_idx}"),
            task_type="wchw_evaluation",
            question=problem["question"],
            metadata={
                "dataset": "WCHW",
                "problem_index": problem_idx,
                "total_problems": len(self.problems)
            }
        )
    
    def evaluate_response(self, task_id: str, response: str) -> A2AResult:
        """Evaluate a purple agent's response"""
        # Find the corresponding problem
        problem = None
        for p in self.problems:
            if p.get("id", "") == task_id:
                problem = p
                break
        
        if problem is None:
            return A2AResult(
                task_id=task_id,
                status="error",
                details={"error": "Problem not found"}
            )
        
        # Use benchmark evaluator
        try:
            score = self.benchmark.evaluator.evaluate(
                question=problem["question"],
                prediction=response,
                answer=problem["answer"]
            )
            
            return A2AResult(
                task_id=task_id,
                status="success",
                answer=response,
                score=score,
                details={
                    "ground_truth": problem["answer"],
                    "prediction": response
                }
            )
        except Exception as e:
            logger.error(f"Evaluation error for task {task_id}: {e}")
            return A2AResult(
                task_id=task_id,
                status="error",
                details={"error": str(e)}
            )
    
    def get_all_tasks(self) -> List[A2ATask]:
        """Get all evaluation tasks"""
        return [self.create_task(i) for i in range(len(self.problems))]
    
    def compute_final_score(self) -> Dict[str, Any]:
        """Compute final benchmark score"""
        if not self.results:
            return {"error": "No results to compute"}
        
        scores = [r.score for r in self.results.values() if r.status == "success"]
        
        return {
            "total_problems": len(self.problems),
            "evaluated": len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "success_rate": len(scores) / len(self.problems) if self.problems else 0
        }


class A2ARequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for A2A Protocol"""
    
    green_agent: WCHWGreenAgent = None
    
    def _set_headers(self, status_code: int = 200, content_type: str = "application/json"):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def _send_json(self, data: Dict, status_code: int = 200):
        self._set_headers(status_code)
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def do_OPTIONS(self):
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            self._send_json({"status": "healthy", "agent": "WirelessAgent Green Agent"})
        
        elif self.path == "/agent-card":
            # A2A Agent Card
            self._send_json({
                "name": "WirelessAgent Green Agent",
                "description": "Evaluation agent for WCHW (Wireless Communication Homework) benchmark",
                "version": "1.0.0",
                "protocol": "A2A",
                "capabilities": ["evaluation", "scoring"],
                "benchmark": {
                    "name": "WCHW",
                    "total_problems": len(self.green_agent.problems),
                    "topics": ["Channel Capacity", "Modulation", "Coding", "Signal Processing", "Propagation", "Noise Analysis"]
                },
                "endpoints": {
                    "tasks": "/tasks",
                    "submit": "/submit",
                    "results": "/results"
                }
            })
        
        elif self.path == "/tasks":
            # Get all evaluation tasks
            tasks = self.green_agent.get_all_tasks()
            self._send_json({
                "total": len(tasks),
                "tasks": [asdict(t) for t in tasks]
            })
        
        elif self.path == "/results":
            # Get current results
            results = {
                "results": {k: asdict(v) for k, v in self.green_agent.results.items()},
                "summary": self.green_agent.compute_final_score()
            }
            self._send_json(results)
        
        elif self.path.startswith("/task/"):
            # Get specific task
            task_id = self.path.split("/")[-1]
            try:
                idx = int(task_id)
                task = self.green_agent.create_task(idx)
                self._send_json(asdict(task))
            except (ValueError, IndexError) as e:
                self._send_json({"error": str(e)}, 404)
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        
        if self.path == "/submit":
            # Submit answer for evaluation
            task_id = data.get("task_id")
            answer = data.get("answer", "")
            
            if not task_id:
                self._send_json({"error": "task_id required"}, 400)
                return
            
            result = self.green_agent.evaluate_response(task_id, answer)
            self.green_agent.results[task_id] = result
            self._send_json(asdict(result))
        
        elif self.path == "/evaluate-all":
            # Evaluate all submitted answers
            answers = data.get("answers", {})  # {task_id: answer}
            
            results = {}
            for task_id, answer in answers.items():
                result = self.green_agent.evaluate_response(task_id, answer)
                self.green_agent.results[task_id] = result
                results[task_id] = asdict(result)
            
            self._send_json({
                "results": results,
                "summary": self.green_agent.compute_final_score()
            })
        
        elif self.path == "/register-purple-agent":
            # Register a purple agent
            purple_url = data.get("url")
            if purple_url:
                self.green_agent.purple_agent_url = purple_url
                self._send_json({"status": "registered", "url": purple_url})
            else:
                self._send_json({"error": "url required"}, 400)
        
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def log_message(self, format, *args):
        logger.info("%s - %s" % (self.address_string(), format % args))


def run_server(host: str = "0.0.0.0", port: int = 8080):
    """Run the Green Agent server"""
    green_agent = WCHWGreenAgent()
    A2ARequestHandler.green_agent = green_agent
    
    server = HTTPServer((host, port), A2ARequestHandler)
    logger.info(f"Starting WirelessAgent Green Agent on http://{host}:{port}")
    logger.info(f"Loaded {len(green_agent.problems)} WCHW test problems")
    logger.info("A2A Endpoints:")
    logger.info("  GET  /health       - Health check")
    logger.info("  GET  /agent-card   - Agent capabilities")
    logger.info("  GET  /tasks        - Get all evaluation tasks")
    logger.info("  GET  /task/<id>    - Get specific task")
    logger.info("  POST /submit       - Submit answer for evaluation")
    logger.info("  POST /evaluate-all - Evaluate all answers")
    logger.info("  GET  /results      - Get current results")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Green Agent server")
        server.shutdown()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WirelessAgent Green Agent - A2A Evaluation Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--dataset", type=str, default="data/datasets/wchw_test_70.jsonl", help="Path to test dataset")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)
