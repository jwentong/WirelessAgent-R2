# WCNS Round 1 - Basic Prompt
# MCTS will optimize this in subsequent rounds

SOLVE_PROMPT = """You are a 5G network slicing expert. Solve this problem step by step.

**Formulas:**
1. Bandwidth (Proportional Fairness): B = Total / (Users + 1)
   - eMBB: B = 90 / (embb_users + 1), range [6-20] MHz
   - URLLC: B = 30 / (urllc_users + 1), range [1-5] MHz
2. Throughput (Shannon): T = 10 * B * log10(1 + 10^(CQI/10))

**Output format (REQUIRED):**
Slice Type: [eMBB or URLLC]
Bandwidth: [value] MHz
Throughput: [value] Mbps

Problem: """
