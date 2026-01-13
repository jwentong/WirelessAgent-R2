"""
WirelessAgent Green Agent - Agent Tests
UC Berkeley AgentX Competition - AgentBeats Compatible

This module contains tests for the WCHW agent.

Author: Jingwen
Date: 1/13/2026
"""

import pytest
import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import WCHWAgent, WCHWEvaluator


class TestWCHWEvaluator:
    """Tests for the WCHW evaluator"""
    
    @pytest.fixture
    def evaluator(self):
        return WCHWEvaluator()
    
    def test_exact_match(self, evaluator):
        """Test exact string match"""
        score = evaluator.evaluate(
            question="What is 2+2?",
            prediction="4",
            answer="4"
        )
        assert score == 1.0
    
    def test_numeric_exact(self, evaluator):
        """Test numeric exact match"""
        score = evaluator.evaluate(
            question="Calculate SNR",
            prediction="0.585",
            answer="0.585"
        )
        assert score == 1.0
    
    def test_numeric_close(self, evaluator):
        """Test numeric within 1% tolerance"""
        score = evaluator.evaluate(
            question="Calculate capacity",
            prediction="0.584",
            answer="0.585"
        )
        assert score >= 0.9
    
    def test_numeric_with_units(self, evaluator):
        """Test numeric with unit prefix"""
        score = evaluator.evaluate(
            question="Calculate bandwidth",
            prediction="16 kHz",
            answer="16000 Hz"
        )
        assert score >= 0.9
    
    def test_scientific_notation(self, evaluator):
        """Test scientific notation"""
        score = evaluator.evaluate(
            question="Calculate power",
            prediction="5.42e-6",
            answer="5.42e-6"
        )
        assert score == 1.0
    
    def test_scientific_with_times(self, evaluator):
        """Test scientific notation with × symbol"""
        score = evaluator.evaluate(
            question="Calculate value",
            prediction="2.2×10^-8",
            answer="2.2e-8"
        )
        assert score >= 0.9
    
    def test_formula_match(self, evaluator):
        """Test formula comparison"""
        score = evaluator.evaluate(
            question="Derive formula",
            prediction="(A^2 * T)/3",
            answer="(A^2 T)/3"
        )
        assert score >= 0.8
    
    def test_wrong_answer(self, evaluator):
        """Test completely wrong answer"""
        score = evaluator.evaluate(
            question="Calculate X",
            prediction="100",
            answer="1"
        )
        assert score == 0.0


class TestWCHWAgent:
    """Tests for the WCHW agent"""
    
    @pytest.fixture
    def agent(self, tmp_path):
        """Create agent with test dataset"""
        # Create test dataset
        test_data = [
            {"id": "test_1", "question": "What is 2+2?", "answer": "4"},
            {"id": "test_2", "question": "Calculate C=B*log2(1+SNR) with B=1MHz, SNR=0.5", "answer": "0.585 Mbit/s"},
            {"id": "test_3", "question": "What is the Shannon limit?", "answer": "C = B log2(1 + SNR)"}
        ]
        
        dataset_path = tmp_path / "test_dataset.jsonl"
        with open(dataset_path, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        return WCHWAgent(dataset_path=str(dataset_path))
    
    def test_load_dataset(self, agent):
        """Test dataset loading"""
        assert len(agent.problems) == 3
    
    def test_get_all_tasks(self, agent):
        """Test getting all tasks"""
        tasks = agent.get_all_tasks()
        assert len(tasks) == 3
        assert tasks[0]["task_id"] == "test_1"
    
    def test_get_task(self, agent):
        """Test getting specific task"""
        task = agent.get_task("test_1")
        assert task is not None
        assert task["question"] == "What is 2+2?"
    
    def test_evaluate_correct_answer(self, agent):
        """Test evaluating correct answer"""
        result = agent.evaluate_response("test_1", "4")
        assert result["status"] == "success"
        assert result["score"] == 1.0
        assert result["passed"] == True
    
    def test_evaluate_wrong_answer(self, agent):
        """Test evaluating wrong answer"""
        result = agent.evaluate_response("test_1", "5")
        assert result["status"] == "success"
        assert result["score"] == 0.0
        assert result["passed"] == False
    
    def test_evaluate_unknown_task(self, agent):
        """Test evaluating unknown task"""
        result = agent.evaluate_response("unknown_task", "answer")
        assert result["status"] == "error"
    
    def test_summary(self, agent):
        """Test getting summary"""
        agent.evaluate_response("test_1", "4")
        agent.evaluate_response("test_2", "0.585 Mbit/s")
        
        summary = agent.get_summary()
        assert summary["completed"] == 2
        assert summary["average_score"] > 0
    
    def test_reset(self, agent):
        """Test resetting results"""
        agent.evaluate_response("test_1", "4")
        agent.reset()
        
        summary = agent.get_summary()
        assert summary["completed"] == 0


@pytest.fixture
def agent_url():
    """Get agent URL from command line or default"""
    return pytest.config.getoption("--agent-url", default="http://localhost:9009")


def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9009",
        help="URL of the running agent to test"
    )


class TestA2AConformance:
    """A2A protocol conformance tests"""
    
    @pytest.fixture
    def agent_url(self, request):
        return request.config.getoption("--agent-url")
    
    @pytest.mark.asyncio
    async def test_agent_card(self, agent_url):
        """Test agent card endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{agent_url}/.well-known/agent.json") as resp:
                assert resp.status == 200
                card = await resp.json()
                
                # Required fields
                assert "name" in card
                assert "description" in card
                assert "skills" in card
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, agent_url):
        """Test health check endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{agent_url}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data.get("status") == "healthy"
    
    @pytest.mark.asyncio
    async def test_tasks_send(self, agent_url):
        """Test tasks/send method"""
        request = {
            "jsonrpc": "2.0",
            "method": "tasks/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "Hello"}]
                }
            },
            "id": "1"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                agent_url,
                json=request,
                headers={"Content-Type": "application/json"}
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                
                assert "result" in data or "error" in data
                if "result" in data:
                    assert "id" in data["result"]
                    assert "status" in data["result"]
    
    @pytest.mark.asyncio
    async def test_assessment_flow(self, agent_url):
        """Test full assessment flow"""
        async with aiohttp.ClientSession() as session:
            # Start assessment
            start_request = {
                "jsonrpc": "2.0",
                "method": "assessment/start",
                "params": {},
                "id": "1"
            }
            
            async with session.post(
                agent_url,
                json=start_request,
                headers={"Content-Type": "application/json"}
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                
                if "result" in data:
                    result = data["result"]
                    assert "session_id" in result or "tasks" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
