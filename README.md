<p align="center">
  <img src="assets/system_overview.png" width="90%" alt="WirelessAgent++ System Overview">
</p>

<h1 align="center">WirelessAgent++</h1>
<h3 align="center">Automated Agentic Workflow Design and Benchmarking for Wireless Networks</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/2603.00501"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv"></a>
  <a href="#wirelessbench"><img src="https://img.shields.io/badge/🏆_WirelessBench-3,392_Problems-blue" alt="WirelessBench"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <a href="https://github.com/jwentong/WirelessAgent-R2/issues"><img src="https://img.shields.io/badge/Issues-Welcome-yellow" alt="Issues"></a>
  <a href="https://github.com/jwentong/WirelessAgent-R2/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs"></a>
</p>

<p align="center">
  <b>WirelessAgent++</b> automates the design of LLM-based autonomous agents for wireless network tasks.<br>
  It casts agent design as a <i>program search problem</i> and solves it with domain-adapted <b>Monte Carlo Tree Search (MCTS)</b>,<br>
  autonomously discovering workflows that outperform hand-crafted baselines by up to <b>31%</b>.
</p>

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [System Overview](#-system-overview)
- [MCTS Optimization](#-mcts-optimization)
- [WirelessBench](#-wirelessbench)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)

---

## ✨ Highlights

| | |
|---|---|
| 🔄 **Automated Agent Design** | No manual workflow engineering — MCTS automatically discovers optimal operator compositions, prompt strategies, and tool-calling patterns |
| 🛠️ **Domain-Specific Tool Integration** | ReAct-based `ToolAgent` and deterministic `CodeLevel` operators seamlessly integrate ray-tracing predictors, Kalman filters, and telecom calculators |
| 📊 **WirelessBench Suite** | 3,392 problems across 3 dimensions: knowledge reasoning (WCHW), network slicing (WCNS), and mobile service assurance (WCMSA) |
| 💰 **Ultra-Low Cost** | Full optimization search costs **< \$5** per task; per-problem inference costs **< \$0.001** |
| 🏆 **State-of-the-Art** | Outperforms prompting baselines by up to **31 pp** and the best general-purpose workflow optimizer by **11.1 pp** |

---

## 🏗️ System Overview

<p align="center">
  <img src="assets/system_overview.png" width="88%" alt="From WirelessAgent (manual) to WirelessAgent++ (automated)">
</p>

**Left:** In WirelessAgent, a human expert iteratively designs a fixed agentic workflow through multi-round dialogue with an LLM.  
**Right:** In WirelessAgent++, an **Optimizer LLM** (Claude-Opus-4.5) jointly searches over workflow structures and tool-calling strategies via MCTS; the resulting workflow is executed by an **Executor LLM** (Qwen-turbo-latest) on WirelessBench, automatically producing distinct, task-adaptive workflows without manual engineering.

### Key Operators

| Operator | Description |
|----------|-------------|
| `Custom(x, p)` | Invokes an LLM with input `x` and instruction prompt `p` |
| `ToolAgent(x, s)` | ReAct-based agent that interleaves reasoning and tool calls for up to `s` iterations |
| `CodeLevel(x, f)` | **LLM-free** deterministic tool execution — zero variance, near-zero cost |
| `ScEnsemble({yᵢ}, x)` | Self-consistency voting across multiple candidate answers |
| `AnswerGenerate(x)` | Structured final answer production |

> 💡 **Key Finding:** The MCTS optimizer autonomously discovers that `ToolAgent`-based workflows can be *compiled* into more efficient `CodeLevel` pipelines, effectively removing the LLM from the tool-calling path while maintaining accuracy.

---

## 🌲 MCTS Optimization

<p align="center">
  <img src="assets/MCTS_Architecture.png" width="80%" alt="MCTS Architecture">
</p>

WirelessAgent++ employs three domain-aware enhancements to the standard MCTS algorithm:

1. **Penalized Boltzmann Selection** — Prevents the optimizer from being trapped in poorly performing subtrees by applying a temperature-controlled penalty to already-visited nodes.

2. **Maturity-Aware Heuristic Critic** — A lightweight LLM pre-screens proposed mutations before expensive evaluation, filtering out obviously poor candidates.

3. **3-Class Experience Replay** — Classifies mutation outcomes as *Success*, *Neutral*, or *Failure* (rather than binary), preventing noise-induced fluctuations from corrupting the experience buffer.

### Optimization Results

- **19 rounds** of WCHW optimization, **\$4.95** total search cost, **~63 min** wall-clock time
- WCHW validation score improved from **62.44%** (Round 1, seed) to **81.78%** (Round 14, best), a **+19.34 pp** gain
- The optimizer discovers tool integration (Round 2, +18.42 pp) as the single largest gain

The held-out test scores reported in the revised manuscript are **78.37%** (WCHW),
**90.95%** (WCNS), and **97.07%** (WCMSA). These values must not be confused with
the validation-set workflow trajectories below or with the historical WCHW “After”
value in the comparison table.

---

## 🏆 WirelessBench

WirelessBench is a standardized, multi-dimensional benchmark suite for evaluating LLM agents on wireless communication tasks:

| Benchmark | Problems | Val / Test | Task Type | Key Challenge |
|-----------|----------|------------|-----------|---------------|
| **WCHW** | 1,392 | 348 / 1,044 | Knowledge Reasoning | Multi-step formula application, unit conversion |
| **WCNS** | 1,000 | 250 / 750 | Code + Tool Use | Ray-tracing CQI prediction → bandwidth allocation |
| **WCMSA** | 1,000 | 250 / 750 | Multi-Step Decision | Kalman prediction → CQI estimation → QoS assurance |

### Data Construction Pipeline

1. **Data Collection** — Seed problems from wireless textbooks (Goldsmith, Molisch) and 3GPP/IEEE standards
2. **Psychometric Data Cleaning** — 10-LLM funnel pipeline with item-total correlation, Mokken scale analysis, and inter-item consistency
3. **LLM-Based Augmentation** — Parameter variation, bidirectional conversion, cross-topic integration (LLMs generate problem text only; all ground truths computed by deterministic solvers)
4. **Human Validation** — Graduate-student verification of every problem

### Workflow Evolution Examples

**WCNS (Network Slicing) — validation-set 3-phase trajectory:**
- **Phase 1 — Seed** (61.3%): Bare LLM call, CQI prediction essentially random
- **Phase 2 — Tool Discovery** (90.5%, +29.2 pp): `ToolAgent` discovers ray-tracing tool
- **Phase 3 — Tool Compilation** (92.18%, +1.7 pp): `ToolAgent` → `CodeLevelRayTracing` (deterministic, LLM-free)

**WCMSA (Mobile Service Assurance) — validation-set 3-phase trajectory:**
- **Phase 1 — Seed** (65.76%): No position prediction or channel estimation
- **Phase 2 — Multi-Tool Discovery** (93.59%, +27.83 pp): `ToolAgent` chains Kalman filter → ray-tracing
- **Phase 3 — Tool Compilation** (96.89%, +1.35 pp): Compiled into `CodeLevelKalmanPredictor` → `CodeLevelRayTracing`

---

## 📈 Results

### WCHW — Wireless Communication Homework

<p align="center">
  <img src="assets/wchw_results.png" width="75%" alt="WCHW Method Comparison">
</p>

### WCNS — Wireless Communication Network Slicing

<p align="center">
  <img src="assets/wcns_results.png" width="75%" alt="WCNS Full Comparison">
</p>

### WCMSA — Mobile Service Assurance

<p align="center">
  <img src="assets/wcmsa_results.png" width="75%" alt="WCMSA Overall Score">
</p>

### Main Results

### Revised-manuscript WirelessBench test results

| Benchmark | Held-out test score |
|-----------|:------------------:|
| WCHW | **78.37%** |
| WCNS | **90.95%** |
| WCMSA | **97.07%** |

The phase scores listed in the WirelessBench section are validation scores used to
describe workflow evolution; they are not replacements for these held-out test
scores.

| Method | HotpotQA (F1) | DROP (F1) | MATH (Solve Rate) | WCHW Before | WCHW After |
|--------|:---:|:---:|:---:|:---:|:---:|
| Qwen-turbo-latest | 0.1423 | 0.5641 | 0.6412 | 0.5013 | 0.5834 |
| CoT | 0.5379 | 0.7607 | 0.6828 | 0.5032 | 0.6032 |
| MedPrompt | 0.5411 | 0.7559 | 0.6996 | 0.5422 | 0.6122 |
| ADAS | 0.5110 | 0.7423 | 0.4953 | 0.4813 | 0.5313 |
| AFlow | 0.5823 | 0.7811 | 0.7864 | 0.5152 | 0.6992 |
| WirelessAgent | -- | -- | -- | 0.6921 | 0.7649 |
| **WirelessAgent++ (Ours)** | **0.7273** | **0.8021** | **0.8210** | **0.7164** | **0.8102** |

### Search Cost

| | WCHW | WCNS | WCMSA |
|---|:---:|:---:|:---:|
| Search Rounds | 19 | 11 | 11 |
| Wall-Clock Time | 63 min | 13 min | 14 min |
| Total Search Cost | $4.95 | $0.99 | $1.05 |
| Per-Problem Inference | $0.00083 | $0.00056 | $0.00068 |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/jwentong/WirelessAgent-R2.git
cd WirelessAgent-R2

# Create conda environment
conda create -n wirelessagent python=3.9
conda activate wirelessagent

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example config and fill in your API keys:

```bash
cp config/config2.example.yaml config/config2.yaml
```

Edit `config/config2.yaml` with your LLM API credentials:

```yaml
models:
  "Claude-Opus-4.5":
    api_type: "openai"
    base_url: "<your_base_url>"
    api_key: "<your_api_key>"
    temperature: 0
  "qwen-turbo-latest":
    api_type: "openai"
    base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: "<your_api_key>"
    temperature: 0
```

### 3. Download Datasets

```bash
python data/download_data.py
```

### 4. Run Optimization

```bash
# Run on WirelessBench benchmarks
python run.py --dataset WCHW --max_rounds 20
python run.py --dataset WCNS --max_rounds 15
python run.py --dataset WCMSA --max_rounds 15

# Run on general NLP benchmarks
python run.py --dataset MATH --max_rounds 20
python run.py --dataset HotpotQA --max_rounds 20
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *(required)* | Benchmark name (WCHW / WCNS / WCMSA / MATH / HotpotQA / DROP / ...) |
| `--sample` | 4 | Number of workflows to resample per round |
| `--max_rounds` | 20 | Maximum MCTS optimization rounds |
| `--initial_round` | 1 | Starting round number |
| `--check_convergence` | False | Enable early stopping |
| `--validation_rounds` | 5 | Validation runs per candidate evaluation |
| `--optimized_path` | auto | Path to save optimized workflows |

---

## 📁 Project Structure

```
WirelessAgent-R2/
├── run.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── config/
│   ├── config2.example.yaml        # API config template
│   └── config2.yaml                # Your API keys (gitignored)
├── benchmarks/
│   ├── benchmark.py                # Base benchmark class
│   ├── wchw.py                     # WCHW benchmark
│   ├── wchw_enhanced.py            # Enhanced WCHW with psychometric cleaning
│   ├── wcns.py                     # WCNS benchmark (network slicing)
│   ├── wcmsa.py                    # WCMSA benchmark (mobile service)
│   ├── hotpotqa.py / drop.py / ... # General NLP benchmarks
│   └── utils.py                    # Benchmark utilities
├── scripts/
│   ├── optimizer.py                # MCTS optimizer core
│   ├── operators.py                # Operator definitions (Custom, ToolAgent, CodeLevel, ...)
│   ├── workflow.py                 # Workflow execution engine
│   ├── evaluator.py                # Evaluation pipeline
│   ├── wireless_tools.py           # Domain tools (ray-tracing, Kalman filter)
│   ├── enhanced_tools.py           # Extended tool library
│   ├── tools.py                    # General tool utilities
│   ├── async_llm.py                # Async LLM API wrapper
│   ├── optimizer_utils/            # MCTS utilities (selection, critic, experience)
│   ├── prompts/                    # Prompt templates
│   ├── rag/                        # RAG retriever for telecom knowledge
│   ├── telecom_tools/              # Telecom-specific tools
│   └── utils/                      # General utilities
├── data/
│   ├── download_data.py            # Dataset downloader
│   ├── maps/                       # HKUST campus ray-tracing maps (.osm)
│   ├── datasets/                   # Downloaded benchmark data (gitignored)
│   └── Textbooks/                  # Reference textbooks (gitignored)
├── assets/                         # Images for README
├── figures/                        # Generated analysis figures
└── workspace/                      # Optimization outputs (gitignored)
```

---

## 📖 Citation

If you find WirelessAgent++ useful in your research, please cite our papers:

```bibtex
@article{tong2026wirelessagentplus,
  title     = {WirelessAgent++: Automated Agentic Workflow Design and Benchmarking for Wireless Networks},
  author    = {Tong, Jingwen and Li, Zijian and Liu Fang and Guo, Wei and Zhang, Jun},
  journal   = {arXiv preprint arXiv:2603.00501v1},
  year      = {2026},
}

@article{tong2025wirelessagent,
  title={WirelessAgent: Large language model agents for intelligent wireless networks},
  author={Tong, Jingwen and Guo, Wei and Shao, Jiawei and Wu, Qiong and Li, Zijian and Lin, Zehong and Zhang, Jun},
  journal={arXiv preprint arXiv:2505.01074},
  year={2025}
}
```

---

## 🙏 Acknowledgments

WirelessAgent++ builds upon the excellent [AFlow](https://github.com/FoundationAgents/AFlow) framework. We thank the AFlow team for their pioneering work on MCTS-based workflow optimization.

This work was supported by the Hong Kong University of Science and Technology (HKUST).

---

<p align="center">
  <i>From "building agents" to "building agent builders" for next-generation wireless networks.</i>
</p>
