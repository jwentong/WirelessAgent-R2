# WirelessAgent

<p align="center">
  <img src="assets/method_comparison.png" alt="Method Comparison" width="700">
</p>

[![Test and Publish](https://github.com/jwentong/WirelessAgent-R2/actions/workflows/test-and-publish.yml/badge.svg)](https://github.com/jwentong/WirelessAgent-R2/actions)
[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2Fjwentong%2Fwirelessagent--r2-blue)](https://github.com/jwentong/WirelessAgent-R2/pkgs/container/wirelessagent-r2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WirelessAgent** is a **Green Agent** for the [UC Berkeley AgentX Competition](https://agentbeats.dev). It evaluates LLM capabilities on wireless communication problems using the **WCHW (Wireless Communication Homework)** benchmark.

---

## 📋 Abstract

WirelessAgent provides a rigorous evaluation framework for assessing LLM agents on 449 wireless communication problems across 6 core topics: Channel Capacity, Modulation, Coding, Signal Processing, Propagation, and Noise Analysis.

The green agent evaluates purple agents using the **A2A (Agent-to-Agent) protocol**, measuring accuracy on numeric calculations, formula derivations, and conceptual understanding. Our baseline achieves **77.94% accuracy**, significantly outperforming standard prompting methods.

---

## 🏗️ Project Structure

```
WirelessAgent-R2/
├── src/
│   ├── server.py         # Server setup and agent card configuration
│   ├── executor.py       # A2A request handling
│   ├── agent.py          # WCHW agent implementation
│   └── messenger.py      # A2A messaging utilities
├── tests/
│   └── test_agent.py     # Agent tests
├── data/datasets/
│   ├── wchw_validate.jsonl  # 349 validation problems
│   └── wchw_test.jsonl      # 100 test problems
├── Dockerfile            # Docker configuration
├── pyproject.toml        # Python dependencies
└── .github/workflows/
    └── test-and-publish.yml  # CI workflow
```

---

## 🚀 Getting Started

### Option 1: Using Docker (Recommended)

```bash
# Pull the image
docker pull ghcr.io/jwentong/wirelessagent-r2:latest

# Run the container
docker run -p 9009:9009 ghcr.io/jwentong/wirelessagent-r2:latest
```

### Option 2: Build from Source

```bash
# Clone the repository
git clone https://github.com/jwentong/WirelessAgent-R2.git
cd WirelessAgent-R2

# Build the image
docker build -t wirelessagent .

# Run the container
docker run -p 9009:9009 wirelessagent
```

### Option 3: Running Locally with uv

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

### Option 4: Running Locally with pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python src/server.py
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# Install test dependencies
uv sync --extra test

# Run tests
uv run pytest tests/test_agent.py -v
```

### Run A2A Conformance Tests

```bash
# Start the agent first
docker run -d -p 9009:9009 --name wirelessagent ghcr.io/jwentong/wirelessagent-r2:latest

# Run conformance tests
uv run pytest tests/test_agent.py --agent-url http://localhost:9009 -v
```

---

## 🔌 A2A Protocol

### Agent Card

```bash
curl http://localhost:9009/.well-known/agent.json
```

```json
{
  "name": "WirelessAgent",
  "description": "A green agent for evaluating LLM capabilities on wireless communication problems",
  "version": "1.0.0",
  "protocol": "A2A",
  "skills": [{
    "id": "wchw-evaluation",
    "name": "WCHW Benchmark Evaluation",
    "description": "Evaluates agents on 100 wireless communication problems"
  }]
}
```

### Assessment Flow

```bash
# 1. Start assessment session
curl -X POST http://localhost:9009 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "assessment/start",
    "params": {},
    "id": "1"
  }'

# 2. Submit answers
curl -X POST http://localhost:9009 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "assessment/submit",
    "params": {
      "session_id": "<session_id>",
      "answers": {
        "test_1": "0.585 Mbit/s",
        "test_2": "16 kHz"
      }
    },
    "id": "2"
  }'

# 3. Get results
curl -X POST http://localhost:9009 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "assessment/results",
    "params": {"session_id": "<session_id>"},
    "id": "3"
  }'
```

### Legacy Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/agent-card` | GET | Agent capabilities |
| `/tasks` | GET | Get all assessment tasks |
| `/submit` | POST | Submit answer for evaluation |
| `/results` | GET | Get current scores |

---

## 📊 WCHW Benchmark

### Dataset Overview

| Split | Problems | Purpose |
|-------|----------|---------|
| Validation | 349 | Workflow optimization |
| Test | 100 | Final evaluation |

<p align="center">
  <img src="assets/knowledge_points_pie.png" alt="Knowledge Points Distribution" width="500">
</p>

### Topics Covered

| Topic | Examples |
|-------|----------|
| Channel Capacity | Shannon capacity, SNR calculations |
| Modulation | ASK, PSK, QAM, FM/PM analysis |
| Coding | Linear block codes, parity bits |
| Signal Processing | Bandwidth, sampling, spectral efficiency |
| Propagation | Free-space loss, path loss models |
| Noise Analysis | BER, noise power, Q-function |

### Scoring

| Error Range | Score |
|-------------|-------|
| < 1% | 1.0 |
| < 5% | 0.9 |
| < 10% | 0.7 |
| < 20% | 0.5 |
| ≥ 20% | 0.0 |

---

## 📈 Results

### Baseline Comparison (Test Set)

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Original (Qwen-Turbo) | 58.34% | - |
| CoT (Wei et al., 2022) | 60.32% | +1.98% |
| MedPrompt (Nori et al., 2023) | 61.22% | +2.88% |
| AFlow (Zhang et al., 2025) | 69.92% | +11.58% |
| **WirelessAgent (Ours)** | **77.94%** | **+19.60%** |

<p align="center">
  <img src="assets/mcts_tree.png" alt="MCTS Tree" width="800">
</p>

---

## 🏆 AgentBeats Registration

### Prerequisites

1. ✅ Docker image published to GitHub Container Registry
2. ✅ A2A protocol endpoints implemented
3. ⏳ Register on [agentbeats.dev](https://agentbeats.dev)

### Steps

1. **Register Green Agent**: Login to agentbeats.dev → Register Agent
   - Display name: `WirelessAgent`
   - Docker image: `ghcr.io/jwentong/wirelessagent-r2:latest`

2. **Create Leaderboard**: Use the [leaderboard template](https://github.com/agentbeats/leaderboard-template)

3. **Connect Leaderboard**: Add webhook URL to your leaderboard repository

---

## 📁 Publishing

The repository includes a GitHub Actions workflow that automatically:
- Runs tests on every push/PR
- Builds and publishes Docker image to GHCR
- Creates releases with version tags

```bash
# Publish latest
git push origin main

# Publish version
git tag v1.0.0
git push origin v1.0.0
```

Images are published to:
- `ghcr.io/jwentong/wirelessagent-r2:latest`
- `ghcr.io/jwentong/wirelessagent-r2:1.0.0`

---

## 📝 License

MIT License

---

## 👤 Author

**Jingwen Tong**  
GitHub: [@jwentong](https://github.com/jwentong)

---

## 🙏 Acknowledgments

- UC Berkeley AgentX Competition
- AgentBeats Platform
- AFlow framework for MCTS optimization methodology
