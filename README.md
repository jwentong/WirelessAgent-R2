# WirelessAgent: LLM Agents for Wireless Communication Tasks

<p align="center">
  <img src="assets/method_comparison.png" alt="Method Comparison" width="700">
</p>

## 📋 Abstract

**WirelessAgent** is a **Green Agent** for evaluating LLM capabilities on wireless communication problems. It provides the **WCHW (Wireless Communication Homework)** benchmark containing 449 problems across 6 core topics: Channel Capacity, Modulation, Coding, Signal Processing, Propagation, and Noise Analysis.

The green agent evaluates purple agents using the A2A (Agent-to-Agent) protocol, measuring accuracy on numeric calculations, formula derivations, and conceptual understanding. Our baseline purple agent achieves **77.94% accuracy** using MCTS-optimized workflows, significantly outperforming standard prompting methods.

---

## 🏆 UC Berkeley AgentX Competition Submission

| Requirement | Status | Description |
|-------------|--------|-------------|
| ✅ Abstract | Complete | Brief description above |
| ✅ GitHub Repository | Complete | This repository with full source code |
| ✅ Baseline Purple Agent | Complete | `agents/purple_agent.py` - A2A compatible |
| ✅ Docker Image | Complete | `Dockerfile` + `docker-compose.yml` |
| ⏳ AgentBeats Registration | Pending | Register on platform |
| ⏳ Demo Video | Pending | 3-minute demonstration |

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/jwentong/WirelessAgent-R2.git
cd WirelessAgent-R2

# Set API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Run with Docker Compose
docker-compose up -d

# Green Agent: http://localhost:8080
# Purple Agent: http://localhost:8081
```

### Option 2: Local Installation

```bash
# Create environment
conda create -n wirelessagent python=3.9
conda activate wirelessagent

# Install dependencies
pip install -r requirements.txt

# Run Green Agent (evaluation server)
python agents/green_agent.py --port 8080

# Run Purple Agent (baseline) in another terminal
python agents/purple_agent.py --port 8081
```

---

## 🟢 Green Agent (Evaluation Server)

The Green Agent evaluates purple/competition agents on the WCHW benchmark.

### A2A Protocol Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/agent-card` | GET | Agent capabilities and metadata |
| `/tasks` | GET | Get all 100 test problems |
| `/task/<id>` | GET | Get specific problem |
| `/submit` | POST | Submit answer for evaluation |
| `/evaluate-all` | POST | Batch evaluate all answers |
| `/results` | GET | Get current scores and summary |

### Example Usage

```bash
# Get agent card
curl http://localhost:8080/agent-card

# Get all tasks
curl http://localhost:8080/tasks

# Submit an answer
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test_1", "answer": "0.585 Mbit/s"}'

# Get results
curl http://localhost:8080/results
```

### Agent Card Response

```json
{
  "name": "WirelessAgent Green Agent",
  "description": "Evaluation agent for WCHW benchmark",
  "version": "1.0.0",
  "protocol": "A2A",
  "capabilities": ["evaluation", "scoring"],
  "benchmark": {
    "name": "WCHW",
    "total_problems": 100,
    "topics": ["Channel Capacity", "Modulation", "Coding", "Signal Processing", "Propagation", "Noise Analysis"]
  }
}
```

---

## 🟣 Purple Agent (Baseline)

The Purple Agent demonstrates how to solve WCHW problems. It uses Chain-of-Thought prompting with an LLM.

### A2A Protocol Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/agent-card` | GET | Agent capabilities |
| `/solve` | POST | Solve a single problem |
| `/run-benchmark` | POST | Run full benchmark against green agent |

### Example Usage

```bash
# Solve a problem
curl -X POST http://localhost:8081/solve \
  -H "Content-Type: application/json" \
  -d '{"question": "Calculate channel capacity with SNR=0.5 and B=1 MHz"}'

# Run full benchmark
curl -X POST http://localhost:8081/run-benchmark \
  -H "Content-Type: application/json" \
  -d '{"green_agent_url": "http://localhost:8080"}'
```

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

<p align="center">
  <img src="assets/knowledge_points_bar.png" alt="Knowledge Points Bar Chart" width="700">
</p>

### Answer Types

| Type | Example | Scoring Method |
|------|---------|----------------|
| Numeric with units | `16 kbit/s`, `44.8 kHz` | Relative error |
| Scientific notation | `5.42e-6`, `2.2×10^-8` | Relative error |
| Mathematical formulas | `(A^2 T)/3`, `1/(2τ_0)` | Symbolic matching |
| LaTeX expressions | `$s_{FM}(t)=3\cos...$` | Pattern matching |

### Scoring Thresholds

| Error Range | Score |
|-------------|-------|
| < 1% | 1.0 |
| < 5% | 0.9 |
| < 10% | 0.7 |
| < 20% | 0.5 |
| ≥ 20% | 0.0 |

---

## 📈 Results

### MCTS Optimization Process

<p align="center">
  <img src="assets/mcts_tree.png" alt="MCTS Tree Visualization" width="800">
</p>

### Performance Evolution

<p align="center">
  <img src="assets/performance_curve.png" alt="Performance Curve" width="700">
</p>

### Baseline Comparison (Test Set)

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Original (Qwen-Turbo) | 58.34% | - |
| ADAS (Hu et al., 2024) | 53.13% | -5.21% |
| CoT-SC (Wang et al., 2022) | 60.01% | +1.67% |
| CoT (Wei et al., 2022) | 60.32% | +1.98% |
| MedPrompt (Nori et al., 2023) | 61.22% | +2.88% |
| AFlow (Zhang et al., 2025) | 69.92% | +11.58% |
| **WirelessAgent (Ours)** | **77.94%** | **+19.60%** |

### Score Distribution

<p align="center">
  <img src="assets/score_analysis.png" alt="Score Distribution" width="700">
</p>

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t wirelessagent:latest .
```

### Run Containers

```bash
# Run Green Agent
docker run -d --name green-agent \
  -p 8080:8080 \
  -v $(pwd)/data:/app/data:ro \
  wirelessagent:latest \
  python agents/green_agent.py

# Run Purple Agent
docker run -d --name purple-agent \
  -p 8081:8081 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  wirelessagent:latest \
  python agents/purple_agent.py
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## 📁 Project Structure

```
WirelessAgent-R2/
├── agents/
│   ├── green_agent.py          # 🟢 Green Agent (A2A evaluation server)
│   ├── purple_agent.py         # 🟣 Purple Agent (baseline solver)
│   └── __init__.py
├── benchmarks/
│   ├── benchmark.py            # Base benchmark class
│   └── wchw.py                 # WCHW evaluation logic
├── scripts/
│   ├── optimizer.py            # MCTS workflow optimizer
│   ├── evaluator.py            # Evaluation orchestrator
│   ├── operators.py            # LLM operators
│   └── telecom_tools/          # Domain-specific tools
├── data/datasets/
│   ├── wchw_validate.jsonl     # 349 validation problems
│   └── wchw_test.jsonl         # 100 test problems
├── config/
│   └── config2.yaml            # LLM configuration
├── assets/                     # Visualization images
├── Dockerfile                  # Docker image definition
├── docker-compose.yml          # Multi-container deployment
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

Edit `config/config2.yaml` to configure LLM providers:

```yaml
models:
  qwen-turbo-latest:
    api_key: "your-api-key"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    temperature: 0.7
  
  gpt-4o-mini:
    api_key: "your-openai-key"
    base_url: "https://api.openai.com/v1"
    temperature: 0.7
```

---

## 🔧 Advanced Usage

### Run MCTS Optimization

```bash
python run.py --dataset WCHW \
    --mode Graph \
    --sample 8 \
    --max_rounds 20 \
    --opt_model_name claude-3-5-sonnet-20241022 \
    --exec_model_name qwen-turbo-latest
```

### Evaluate on Test Set

```bash
python run.py --dataset WCHW --mode Test --test_rounds 14
```

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
- AFlow framework for MCTS optimization methodology
