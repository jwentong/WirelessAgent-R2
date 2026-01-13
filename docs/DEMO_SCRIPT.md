# WirelessAgent Demo Video Script
# UC Berkeley AgentX Competition Submission
# Duration: 3 minutes
# Author: Jingwen Tong

================================================================================
## PREPARATION CHECKLIST
================================================================================

Before recording:
[ ] Close unnecessary applications
[ ] Set VS Code to dark theme with large font (16pt+)
[ ] Open terminal in VS Code
[ ] Have these files ready to show:
    - README.md
    - data/datasets/wchw_test.jsonl
    - agents/green_agent.py
    - assets/method_comparison.png
    - assets/knowledge_points_pie.png
    - assets/mcts_tree.png
[ ] Pre-run: docker-compose up -d (so containers are ready)
[ ] Test all curl commands beforehand

================================================================================
## SCENE 1: INTRODUCTION (0:00 - 0:30)
================================================================================

[SCREEN: Show README.md with project title visible]
[ACTION: Scroll to show the method_comparison.png image]

SCRIPT:
"Hi, I'm Jingwen Tong, and this is WirelessAgent - a Green Agent submission 
for the UC Berkeley AgentX Competition.

WirelessAgent evaluates LLM capabilities on wireless communication problems 
using our WCHW benchmark - that's Wireless Communication Homework.

Our baseline purple agent achieves 77.94% accuracy, significantly outperforming 
existing methods like Chain-of-Thought and AFlow."

================================================================================
## SCENE 2: WCHW DATASET (0:30 - 1:15)
================================================================================

[SCREEN: Show knowledge_points_pie.png]

SCRIPT:
"The WCHW benchmark contains 449 carefully curated problems across 6 core topics:
Channel Capacity, Modulation, Coding, Signal Processing, Propagation, and 
Noise Analysis."

[ACTION: Open data/datasets/wchw_test.jsonl, show a sample problem]

SCRIPT:
"Each problem includes a question, ground-truth answer, and chain-of-thought 
solution. Here's an example - calculating channel capacity with given SNR 
and bandwidth.

The benchmark supports multiple answer types: numeric values with units, 
scientific notation, mathematical formulas, and LaTeX expressions."

[ACTION: Scroll to show 2-3 different problem types]

SCRIPT:
"This diversity makes evaluation challenging - our green agent handles 
automatic unit conversion and formula normalization for accurate scoring."

================================================================================
## SCENE 3: A2A PROTOCOL DEMO (1:15 - 2:15)
================================================================================

[SCREEN: Switch to terminal]

SCRIPT:
"Let me demonstrate the A2A protocol. Our system runs as Docker containers."

[ACTION: Type and run]
```
docker-compose ps
```

SCRIPT:
"The green agent runs on port 8080, and the purple agent on 8081."

[ACTION: Type and run]
```
curl http://localhost:8080/agent-card
```

SCRIPT:
"The agent-card endpoint returns our green agent's capabilities and benchmark 
metadata - 100 test problems across 6 topics."

[ACTION: Type and run]
```
curl http://localhost:8080/tasks | head -30
```

SCRIPT:
"Purple agents fetch tasks from the /tasks endpoint. Each task contains 
a unique ID and the problem question."

[ACTION: Type and run]
```
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{"task_id": "test_1", "answer": "0.585 Mbit/s"}'
```

SCRIPT:
"When a purple agent submits an answer, our green agent evaluates it 
immediately. Here we get a score of 1.0 - a perfect match."

[ACTION: Type and run]
```
curl http://localhost:8080/results
```

SCRIPT:
"The /results endpoint provides a summary of all evaluated answers, 
including average score and success rate."

================================================================================
## SCENE 4: RESULTS (2:15 - 2:45)
================================================================================

[SCREEN: Show method_comparison.png full screen]

SCRIPT:
"Our baseline purple agent uses MCTS-optimized workflows to achieve 
77.94% accuracy on the test set.

This is an 8% improvement over AFlow, and nearly 20% better than 
standard prompting with Qwen-Turbo."

[ACTION: Switch to mcts_tree.png]

SCRIPT:
"The key innovation is our Monte Carlo Tree Search optimization. 
Starting from template workflows, we explore and refine agent strategies 
over multiple rounds, converging to high-performing solutions."

================================================================================
## SCENE 5: CONCLUSION (2:45 - 3:00)
================================================================================

[SCREEN: Show README.md, scroll to GitHub section]

SCRIPT:
"WirelessAgent provides a rigorous benchmark for evaluating LLMs on 
technical wireless communication problems.

The complete source code, Docker images, and baseline agents are 
available on GitHub at github.com/jwentong/WirelessAgent-R2.

Thank you for watching!"

[ACTION: Hold on GitHub link for 3 seconds]

================================================================================
## BACKUP COMMANDS (if needed)
================================================================================

# Start containers
docker-compose up -d

# Check health
curl http://localhost:8080/health
curl http://localhost:8081/health

# Get single task
curl http://localhost:8080/task/0

# Run full benchmark
curl -X POST http://localhost:8081/run-benchmark \
  -H "Content-Type: application/json" \
  -d '{"green_agent_url": "http://green-agent:8080"}'

# Stop containers
docker-compose down

================================================================================
## TIMING NOTES
================================================================================

- Speak at moderate pace (~150 words/minute)
- Pause briefly between sections
- If running short on time, skip the /tasks command output
- If running long, shorten the MCTS explanation

================================================================================
## POST-PRODUCTION
================================================================================

1. Add title card at beginning:
   "WirelessAgent: LLM Agents for Wireless Communication Tasks"
   "UC Berkeley AgentX Competition"
   "Jingwen Tong"

2. Add captions/subtitles (optional but recommended)

3. Add background music (low volume, no lyrics)

4. Export settings:
   - Resolution: 1920x1080 (1080p)
   - Frame rate: 30 fps
   - Format: MP4 (H.264)
   - Max file size: Check competition requirements

================================================================================
