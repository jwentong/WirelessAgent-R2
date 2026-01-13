"""
WirelessAgent Green Agent - A2A Messaging Utilities
UC Berkeley AgentX Competition - AgentBeats Compatible

This module provides utilities for A2A protocol messaging.

Author: Jingwen
Date: 1/13/2026
"""

import json
import aiohttp
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("WirelessAgent.Messenger")


@dataclass
class TextPart:
    """Text message part"""
    type: str = "text"
    text: str = ""


@dataclass
class FilePart:
    """File message part"""
    type: str = "file"
    mimeType: str = ""
    data: str = ""  # Base64 encoded


@dataclass
class A2AMessage:
    """A2A Protocol Message"""
    role: str  # "user" or "agent"
    parts: List[Dict[str, Any]]
    
    @classmethod
    def text(cls, role: str, content: str) -> "A2AMessage":
        """Create a text message"""
        return cls(role=role, parts=[asdict(TextPart(text=content))])
    
    @classmethod
    def from_dict(cls, data: Dict) -> "A2AMessage":
        """Create from dictionary"""
        return cls(
            role=data.get("role", "user"),
            parts=data.get("parts", [])
        )
    
    def get_text(self) -> str:
        """Extract text content from message"""
        for part in self.parts:
            if part.get("type") == "text":
                return part.get("text", "")
        return ""


@dataclass
class A2ARequest:
    """A2A JSON-RPC Request"""
    method: str
    params: Dict[str, Any]
    id: str = None
    jsonrpc: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params,
            "id": self.id or "1"
        }


@dataclass
class A2AResponse:
    """A2A JSON-RPC Response"""
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    id: str = None
    jsonrpc: str = "2.0"
    
    @classmethod
    def from_dict(cls, data: Dict) -> "A2AResponse":
        return cls(
            result=data.get("result"),
            error=data.get("error"),
            id=data.get("id"),
            jsonrpc=data.get("jsonrpc", "2.0")
        )
    
    @property
    def is_error(self) -> bool:
        return self.error is not None


class A2AClient:
    """
    A2A Protocol Client
    
    Sends requests to A2A-compatible agents.
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card from target agent"""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/.well-known/agent.json",
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as resp:
                return await resp.json()
    
    async def send_task(
        self,
        message: A2AMessage,
        session_id: str = None,
        task_id: str = None
    ) -> A2AResponse:
        """Send a task to the agent"""
        request = A2ARequest(
            method="tasks/send",
            params={
                "message": asdict(message),
                "sessionId": session_id,
                "id": task_id
            }
        )
        return await self._send_request(request)
    
    async def get_task(self, task_id: str) -> A2AResponse:
        """Get task status"""
        request = A2ARequest(
            method="tasks/get",
            params={"id": task_id}
        )
        return await self._send_request(request)
    
    async def cancel_task(self, task_id: str) -> A2AResponse:
        """Cancel a task"""
        request = A2ARequest(
            method="tasks/cancel",
            params={"id": task_id}
        )
        return await self._send_request(request)
    
    async def start_assessment(self) -> A2AResponse:
        """Start an assessment session"""
        request = A2ARequest(
            method="assessment/start",
            params={}
        )
        return await self._send_request(request)
    
    async def submit_answers(
        self,
        session_id: str,
        answers: Dict[str, str]
    ) -> A2AResponse:
        """Submit answers for assessment"""
        request = A2ARequest(
            method="assessment/submit",
            params={
                "session_id": session_id,
                "answers": answers
            }
        )
        return await self._send_request(request)
    
    async def get_results(self, session_id: str = None) -> A2AResponse:
        """Get assessment results"""
        request = A2ARequest(
            method="assessment/results",
            params={"session_id": session_id}
        )
        return await self._send_request(request)
    
    async def _send_request(self, request: A2ARequest) -> A2AResponse:
        """Send A2A request"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    self.base_url,
                    json=request.to_dict(),
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    data = await resp.json()
                    return A2AResponse.from_dict(data)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return A2AResponse(
                    error={"code": -32000, "message": str(e)}
                )


def create_assessment_task(task_id: str, question: str) -> Dict[str, Any]:
    """Create an assessment task payload"""
    return {
        "task_id": task_id,
        "question": question,
        "type": "wchw_problem"
    }


def parse_assessment_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Parse assessment result from response"""
    return {
        "task_id": result.get("task_id"),
        "score": result.get("score", 0.0),
        "max_score": result.get("max_score", 1.0),
        "passed": result.get("passed", False),
        "feedback": result.get("details", {})
    }


async def run_assessment_against_agent(
    green_agent_url: str,
    purple_agent_url: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run a full assessment of a purple agent against the green agent.
    
    Args:
        green_agent_url: URL of the green agent (evaluator)
        purple_agent_url: URL of the purple agent (solver)
        timeout: Total timeout in seconds
    
    Returns:
        Assessment results
    """
    green_client = A2AClient(green_agent_url)
    purple_client = A2AClient(purple_agent_url)
    
    # Start assessment
    start_response = await green_client.start_assessment()
    if start_response.is_error:
        return {"error": start_response.error}
    
    session_id = start_response.result.get("session_id")
    tasks = start_response.result.get("tasks", [])
    
    logger.info(f"Started assessment with {len(tasks)} tasks")
    
    # Solve each task with purple agent
    answers = {}
    for task in tasks:
        task_id = task.get("task_id")
        question = task.get("question")
        
        # Send to purple agent
        message = A2AMessage.text("user", question)
        response = await purple_client.send_task(message)
        
        if not response.is_error:
            # Extract answer from response
            history = response.result.get("history", [])
            for msg in reversed(history):
                if msg.get("role") == "agent":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            answers[task_id] = part.get("text", "")
                            break
                    break
    
    # Submit all answers
    submit_response = await green_client.submit_answers(session_id, answers)
    if submit_response.is_error:
        return {"error": submit_response.error}
    
    # Get final results
    results_response = await green_client.get_results(session_id)
    return results_response.result if not results_response.is_error else {"error": results_response.error}
