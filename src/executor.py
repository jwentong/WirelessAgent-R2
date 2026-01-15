"""
WirelessAgent Green Agent - A2A Request Executor
UC Berkeley AgentX Competition - AgentBeats Compatible

This module handles A2A assessment requests and orchestrates the evaluation flow.
The Green Agent actively sends problems to Purple Agent and collects answers.

Author: Jingwen
Date: 1/15/2026
"""

import asyncio
import aiohttp
import logging
import uuid
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger("WirelessAgent.Executor")


class TaskState(str, Enum):
    """A2A Task States"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Message:
    """A2A Message"""
    role: str  # "user" or "agent"
    parts: List[Dict[str, Any]]
    messageId: str = None
    
    def __post_init__(self):
        if self.messageId is None:
            self.messageId = str(uuid.uuid4())
    
    @classmethod
    def text(cls, role: str, content: str) -> "Message":
        return cls(role=role, parts=[{"type": "text", "text": content}], messageId=str(uuid.uuid4()))


@dataclass
class Task:
    """A2A Task"""
    id: str
    sessionId: str
    status: Dict[str, Any]
    contextId: str = None
    history: List[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.contextId is None:
            self.contextId = str(uuid.uuid4())
        if self.history is None:
            self.history = []
        if self.artifacts is None:
            self.artifacts = []


class WCHWExecutor:
    """
    A2A Request Executor for WCHW Assessment
    
    Orchestrates the evaluation flow:
    1. Receive assessment request from AgentBeats client
    2. Connect to Purple Agent and send WCHW problems
    3. Collect answers and compute scores
    4. Return assessment results
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.sessions: Dict[str, Dict] = {}
        self.tasks: Dict[str, Task] = {}
        # Purple Agent endpoint - read from environment or use default
        self.purple_agent_url = os.environ.get("PURPLE_AGENT_URL", "http://purple-agent:9009")
    
    async def execute(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an A2A request"""
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "tasks/send" or method == "message/send":
                result = await self._handle_send(params)
            elif method == "tasks/get" or method == "message/get":
                result = await self._handle_get(params)
            elif method == "tasks/cancel" or method == "message/cancel":
                result = await self._handle_cancel(params)
            elif method == "assessment/start":
                result = await self._handle_assessment_start(params)
            elif method == "assessment/submit":
                result = await self._handle_assessment_submit(params)
            elif method == "assessment/results":
                result = await self._handle_assessment_results(params)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)}
            }

    async def _send_to_purple_agent(self, problem: Dict[str, Any]) -> str:
        """Send a problem to Purple Agent and get the answer"""
        question = problem.get("question", "")
        task_id = problem.get("id", str(uuid.uuid4()))
        
        # Prepare A2A message to Purple Agent
        a2a_request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "id": task_id,
                "sessionId": str(uuid.uuid4()),
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"type": "text", "text": question}]
                }
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.purple_agent_url,
                    json=a2a_request,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Extract answer from response
                        if "result" in result:
                            task_result = result["result"]
                            # Try to get answer from history
                            history = task_result.get("history", [])
                            for msg in reversed(history):
                                if msg.get("role") == "agent":
                                    parts = msg.get("parts", [])
                                    for part in parts:
                                        if part.get("type") == "text":
                                            return part.get("text", "")
                            # Try artifacts
                            artifacts = task_result.get("artifacts", [])
                            for artifact in artifacts:
                                if "answer" in artifact:
                                    return str(artifact["answer"])
                        return ""
                    else:
                        logger.error(f"Purple Agent returned status {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Error communicating with Purple Agent: {e}")
            return ""

    async def _run_assessment(self, purple_agent_url: str = None) -> Dict[str, Any]:
        """Run full WCHW assessment against Purple Agent"""
        if purple_agent_url:
            self.purple_agent_url = purple_agent_url
            
        logger.info(f"Starting WCHW assessment with Purple Agent at {self.purple_agent_url}")
        
        start_time = time.time()
        problems = self.agent.get_all_tasks()
        total_tasks = len(problems)
        
        logger.info(f"Evaluating {total_tasks} problems...")
        
        results = []
        correct = 0
        
        for i, problem in enumerate(problems):
            task_id = problem.get("task_id", str(i))
            question = problem.get("question", "")
            
            # Get answer from Purple Agent
            answer = await self._send_to_purple_agent(problem)
            
            # Evaluate the answer
            eval_result = self.agent.evaluate_response(task_id, answer)
            score = eval_result.get("score", 0)
            
            if score >= 0.5:
                correct += 1
            
            results.append({
                "task_id": task_id,
                "question": question[:100] + "..." if len(question) > 100 else question,
                "predicted_answer": answer[:200] if answer else "",
                "score": score,
                "correct": score >= 0.5
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{total_tasks} ({correct}/{i+1} correct)")
        
        end_time = time.time()
        time_used = end_time - start_time
        
        # Calculate final metrics
        average_score = sum(r["score"] for r in results) / len(results) if results else 0
        accuracy = correct / total_tasks if total_tasks > 0 else 0
        
        summary = {
            "total_tasks": total_tasks,
            "correct": correct,
            "accuracy": accuracy,
            "average_score": average_score,
            "time_used": time_used,
            "avg_time_per_task": time_used / total_tasks if total_tasks > 0 else 0,
            "results": results
        }
        
        logger.info(f"Assessment complete: {correct}/{total_tasks} correct ({accuracy:.2%})")
        logger.info(f"Average score: {average_score:.4f}")
        logger.info(f"Time used: {time_used:.2f}s")
        
        return summary
    
    async def _handle_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/send or message/send - This triggers the assessment"""
        session_id = params.get("sessionId") or str(uuid.uuid4())
        task_id = params.get("id") or str(uuid.uuid4())
        message = params.get("message", {})
        
        # Extract text from message
        text = ""
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text = part.get("text", "")
                break
        
        logger.info(f"Received message: {text[:100] if text else '(empty)'}")
        
        # Run the assessment automatically when receiving any message
        # The AgentBeats client sends a message to trigger assessment
        try:
            # Run full assessment against Purple Agent
            assessment_result = await self._run_assessment()
            
            # Create response task with assessment results
            artifact_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                sessionId=session_id,
                status={"state": TaskState.COMPLETED},
                history=[
                    asdict(Message.text("user", text)),
                    asdict(Message.text("agent", f"WCHW Assessment Complete!\n\n"
                        f"Total Problems: {assessment_result['total_tasks']}\n"
                        f"Correct: {assessment_result['correct']}\n"
                        f"Accuracy: {assessment_result['accuracy']:.2%}\n"
                        f"Average Score: {assessment_result['average_score']:.4f}\n"
                        f"Time Used: {assessment_result['time_used']:.2f}s"))
                ],
                artifacts=[{
                    "artifactId": artifact_id,
                    "name": "assessment_results",
                    "parts": [{
                        "type": "data",
                        "mimeType": "application/json",
                        "data": assessment_result
                    }]
                }]
            )
            
            self.tasks[task_id] = task
            self.sessions[session_id] = {
                "task_id": task_id,
                "results": assessment_result
            }
            
            return asdict(task)
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            import traceback
            traceback.print_exc()
            
            task = Task(
                id=task_id,
                sessionId=session_id,
                status={"state": TaskState.FAILED},
                history=[
                    asdict(Message.text("user", text)),
                    asdict(Message.text("agent", f"Assessment failed: {str(e)}"))
                ]
            )
            self.tasks[task_id] = task
            return asdict(task)
    
    async def _handle_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/get - Get task status"""
        task_id = params.get("id")
        if task_id not in self.tasks:
            raise ValueError(f"Task not found: {task_id}")
        return asdict(self.tasks[task_id])
    
    async def _handle_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/cancel - Cancel a task"""
        task_id = params.get("id")
        if task_id in self.tasks:
            self.tasks[task_id].status = {"state": TaskState.CANCELED}
            return asdict(self.tasks[task_id])
        raise ValueError(f"Task not found: {task_id}")
    
    async def _handle_assessment_start(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle assessment/start - Begin WCHW assessment"""
        session_id = str(uuid.uuid4())
        tasks = self.agent.get_all_tasks()
        
        self.sessions[session_id] = {
            "answers": {},
            "scores": {},
            "total_tasks": len(tasks)
        }
        
        return {
            "session_id": session_id,
            "total_tasks": len(tasks),
            "tasks": tasks,
            "message": "Assessment started. Submit answers using assessment/submit."
        }
    
    async def _handle_assessment_submit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle assessment/submit - Submit answers for evaluation"""
        session_id = params.get("session_id")
        answers = params.get("answers", {})  # {task_id: answer}
        
        if session_id not in self.sessions:
            raise ValueError(f"Session not found: {session_id}")
        
        session = self.sessions[session_id]
        results = {}
        
        for task_id, answer in answers.items():
            result = self.agent.evaluate_response(task_id, answer)
            session["answers"][task_id] = answer
            session["scores"][task_id] = result.get("score", 0)
            results[task_id] = result
        
        # Calculate summary
        scores = list(session["scores"].values())
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "session_id": session_id,
            "submitted": len(answers),
            "total_evaluated": len(session["scores"]),
            "current_average": avg_score,
            "results": results
        }
    
    async def _handle_assessment_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle assessment/results - Get final assessment results"""
        session_id = params.get("session_id")
        
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            scores = list(session["scores"].values())
            
            return {
                "session_id": session_id,
                "total_tasks": session.get("total_tasks", 0),
                "completed_tasks": len(session["scores"]),
                "pass_rate": sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "scores": session["scores"]
            }
        else:
            # Return global summary
            return self.agent.get_summary()
