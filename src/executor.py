"""
WirelessAgent Green Agent - A2A Request Executor
UC Berkeley AgentX Competition - AgentBeats Compatible

This module handles A2A assessment requests and orchestrates the evaluation flow.

Author: Jingwen
Date: 1/13/2026
"""

import asyncio
import logging
import uuid
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
    
    @classmethod
    def text(cls, role: str, content: str) -> "Message":
        return cls(role=role, parts=[{"type": "text", "text": content}])


@dataclass
class Task:
    """A2A Task"""
    id: str
    sessionId: str
    status: Dict[str, Any]
    history: List[Dict[str, Any]] = None
    artifacts: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.artifacts is None:
            self.artifacts = []


class WCHWExecutor:
    """
    A2A Request Executor for WCHW Assessment
    
    Handles the assessment flow:
    1. Receive task from purple agent
    2. Send WCHW problems for evaluation
    3. Receive answers and compute scores
    4. Return assessment results
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.sessions: Dict[str, Dict] = {}
        self.tasks: Dict[str, Task] = {}
    
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
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32603, "message": str(e)}
            }
    
    async def _handle_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tasks/send - Start or continue a task"""
        session_id = params.get("sessionId") or str(uuid.uuid4())
        task_id = params.get("id") or str(uuid.uuid4())
        message = params.get("message", {})
        
        # Extract text from message
        text = ""
        for part in message.get("parts", []):
            if part.get("type") == "text":
                text = part.get("text", "")
                break
        
        # Check if this is an assessment request
        if "start assessment" in text.lower() or "begin evaluation" in text.lower():
            # Start WCHW assessment
            tasks = self.agent.get_all_tasks()
            
            task = Task(
                id=task_id,
                sessionId=session_id,
                status={"state": TaskState.INPUT_REQUIRED},
                history=[
                    asdict(Message.text("user", text)),
                    asdict(Message.text("agent", f"Starting WCHW assessment with {len(tasks)} problems. Please solve each problem and submit your answers."))
                ],
                artifacts=[{
                    "type": "assessment_tasks",
                    "data": tasks
                }]
            )
            
            self.tasks[task_id] = task
            self.sessions[session_id] = {
                "task_id": task_id,
                "current_problem": 0,
                "answers": {},
                "scores": {}
            }
            
            return asdict(task)
        
        # Check if this is an answer submission
        elif task_id in self.tasks:
            session = self.sessions.get(self.tasks[task_id].sessionId, {})
            
            # Parse answer from message
            # Expected format: "task_id: answer" or JSON
            try:
                import json
                answer_data = json.loads(text)
                task_answer_id = answer_data.get("task_id")
                answer = answer_data.get("answer")
            except:
                # Try to parse simple format
                if ":" in text:
                    parts = text.split(":", 1)
                    task_answer_id = parts[0].strip()
                    answer = parts[1].strip()
                else:
                    task_answer_id = None
                    answer = text
            
            if task_answer_id and answer:
                result = self.agent.evaluate_response(task_answer_id, answer)
                session["answers"][task_answer_id] = answer
                session["scores"][task_answer_id] = result.get("score", 0)
                
                # Check if all problems answered
                all_tasks = self.agent.get_all_tasks()
                if len(session["scores"]) >= len(all_tasks):
                    # Assessment complete
                    summary = self.agent.get_summary()
                    self.tasks[task_id].status = {"state": TaskState.COMPLETED}
                    self.tasks[task_id].artifacts.append({
                        "type": "assessment_results",
                        "data": summary
                    })
                    self.tasks[task_id].history.append(
                        asdict(Message.text("agent", f"Assessment complete! Final score: {summary.get('average_score', 0):.2%}"))
                    )
                else:
                    self.tasks[task_id].history.append(
                        asdict(Message.text("agent", f"Answer received for {task_answer_id}. Score: {result.get('score', 0):.2f}. {len(all_tasks) - len(session['scores'])} problems remaining."))
                    )
            
            return asdict(self.tasks[task_id])
        
        else:
            # New task - return welcome message
            task = Task(
                id=task_id,
                sessionId=session_id,
                status={"state": TaskState.COMPLETED},
                history=[
                    asdict(Message.text("user", text)),
                    asdict(Message.text("agent", "Welcome to WirelessAgent WCHW Benchmark. Send 'start assessment' to begin evaluation."))
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
