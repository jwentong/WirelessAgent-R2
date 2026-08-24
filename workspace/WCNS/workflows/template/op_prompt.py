# -*- coding: utf-8 -*-
"""
WCNS (Wireless Communication Network Slicing) Dataset - Operator Prompts
Version v1.0 (2026-01-21)

Prompts for network slicing decision-making workflow:
1. Intent Understanding - Analyze user request to identify service type
2. Slice Allocation - Decide eMBB or URLLC based on requirements
3. Bandwidth Allocation - Calculate optimal bandwidth using Shannon formula
4. QoS Evaluation - Verify requirements are met
5. Adjustment - Check if dynamic adjustment is needed
"""

# ============================================================================
# Self-Consistency Ensemble Prompt
# ============================================================================

SC_ENSEMBLE_PROMPT = """
Given the network slicing problem described as follows: {problem}

Several solutions have been generated to address the given network slicing request. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. Consider:
1. Is the slice type (eMBB vs URLLC) correctly chosen based on the service requirements?
2. Is the bandwidth allocation within the valid range for the chosen slice?
3. Does the expected rate satisfy the QoS requirements?

In the "thought" field, provide a detailed explanation of your evaluation process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent and correct solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

# ============================================================================
# Network Slicing Knowledge Base
# ============================================================================

SLICE_KNOWLEDGE_BASE = """
=== 5G NETWORK SLICING KNOWLEDGE BASE ===

【SLICE TYPES】

1. eMBB (enhanced Mobile Broadband):
   - Use for: Video streaming, AR/VR, file download, web browsing, social media
   - Bandwidth range: 6-20 MHz
   - Rate range: 100-400 Mbps
   - Total capacity: 90 MHz
   - Keywords: video, stream, download, AR, VR, browse, file, cloud, 4K, 8K

2. URLLC (Ultra-Reliable Low-Latency Communications):
   - Use for: Gaming, VoIP, remote control, IoT, safety-critical, real-time
   - Bandwidth range: 1-5 MHz
   - Rate range: 1-100 Mbps
   - Total capacity: 30 MHz
   - Keywords: game, call, control, robot, IoT, sensor, real-time, critical, monitor

【BANDWIDTH ALLOCATION FORMULA】

Shannon Capacity Formula:
Rate = α × Bandwidth × log₁₀(1 + 10^(CQI/10))
where α = 10.0

For CQI values:
- CQI 1-5:   Low quality, need more bandwidth
- CQI 6-10:  Medium quality, standard allocation
- CQI 11-15: High quality, efficient bandwidth use

【PROPORTIONAL FAIRNESS ALLOCATION】

Objective: Maximize Σ log(Rate_i) for all users
Constraints:
- Σ Bandwidth_i ≤ Total_Capacity
- B_min ≤ Bandwidth_i ≤ B_max for each user
- Rate_i must satisfy slice QoS requirements

【QoS REQUIREMENTS BY SERVICE TYPE】

| Service Type    | Slice | Min Rate | Max Latency | Reliability |
|-----------------|-------|----------|-------------|-------------|
| Video Streaming | eMBB  | 25 Mbps  | 100 ms      | 99%         |
| Online Gaming   | URLLC | 5 Mbps   | 20 ms       | 99.9%       |
| VoIP Call       | URLLC | 0.5 Mbps | 50 ms       | 99.9%       |
| Web Browsing    | eMBB  | 5 Mbps   | 200 ms      | 95%         |
| File Download   | eMBB  | 50 Mbps  | 500 ms      | 99%         |
| AR Navigation   | eMBB  | 20 Mbps  | 50 ms       | 99%         |
| Remote Control  | URLLC | 1 Mbps   | 10 ms       | 99.99%      |
| IoT Monitoring  | URLLC | 0.1 Mbps | 100 ms      | 99.9%       |

【DECISION WORKFLOW】

Step 1: Intent Understanding
- Extract keywords from user request
- Query knowledge base for service classification
- Identify primary QoS requirements

Step 2: Slice Type Selection
- High bandwidth need + latency tolerance → eMBB
- Low latency + mission critical → URLLC

Step 3: Bandwidth Calculation
- Get CQI from network state
- Apply Shannon formula to determine rate
- Use proportional fairness for multi-user scenario

Step 4: QoS Verification
- Check: Rate ≥ Min_Rate for service type
- Check: Bandwidth within slice constraints

Step 5: Adjustment (if needed)
- If insufficient capacity: reduce existing user allocation
- If poor channel: increase bandwidth to compensate
"""

# ============================================================================
# Intent Understanding Prompt
# ============================================================================

INTENT_UNDERSTANDING_PROMPT = """You are a 5G network slicing intent analyzer.

Analyze the following user request and network state to determine the service type and requirements.

User Request: {request}
User Position: ({x}, {y}, {z})
Channel Quality Indicator (CQI): {cqi}

Current Network State:
- eMBB Slice: {embb_users} users, {embb_utilization}% utilization
- URLLC Slice: {urllc_users} users, {urllc_utilization}% utilization

Task:
1. Identify the service type (video_streaming, online_gaming, voip_call, web_browsing, file_download, ar_navigation, remote_control, iot_monitoring)
2. Determine the appropriate slice type (eMBB or URLLC)
3. List the QoS requirements (min_rate, max_latency, reliability)

{knowledge_base}

Provide your analysis in a structured format.
"""

# ============================================================================
# Bandwidth Allocation Prompt
# ============================================================================

BANDWIDTH_ALLOCATION_PROMPT = """You are a 5G resource allocation expert.

Based on the service analysis, calculate the optimal bandwidth allocation.

Service Type: {service_type}
Recommended Slice: {slice_type}
User CQI: {cqi}
SNR (dB): {snr_db}

Network State:
- {slice_type} Slice: {used_bandwidth}/{total_bandwidth} MHz used
- Existing users: {num_existing_users}

Slice Constraints:
- Bandwidth range: {bw_min}-{bw_max} MHz
- Rate range: {rate_min}-{rate_max} Mbps

Shannon Capacity Formula:
Rate = 10.0 × Bandwidth × log₁₀(1 + 10^(CQI/10))

Task:
1. Calculate the spectral efficiency: log₁₀(1 + 10^({cqi}/10))
2. Determine optimal bandwidth considering:
   - Available capacity
   - Proportional fairness with existing users
   - QoS requirements
3. Calculate expected rate with allocated bandwidth

Show your calculations step by step.
"""

# ============================================================================
# QoS Evaluation Prompt
# ============================================================================

QOS_EVALUATION_PROMPT = """You are a QoS evaluation expert.

Evaluate if the allocated resources satisfy the service requirements.

Service Type: {service_type}
Allocated Bandwidth: {allocated_bandwidth} MHz
Expected Rate: {expected_rate} Mbps

QoS Requirements:
- Minimum Rate: {min_rate} Mbps
- Maximum Latency: {max_latency} ms
- Reliability: {reliability}%

Slice Constraints:
- Bandwidth range: {bw_min}-{bw_max} MHz

Check:
1. Is the expected rate ≥ minimum required rate?
2. Is the bandwidth within the slice constraints?
3. Is the slice type appropriate for the latency requirement?

Provide evaluation result and any recommended adjustments.
"""

# ============================================================================
# ReAct Strategy Prompt - WCNS Optimized
# ============================================================================

REACT_STRATEGY_PROMPT = """You are a 5G network slicing expert. Your task is to determine the optimal slice type and bandwidth allocation for a new user.

=== CRITICAL: PROPORTIONAL FAIRNESS ALGORITHM ===

**This is the KEY algorithm for bandwidth allocation!**

The bandwidth for a new user is calculated using PROPORTIONAL FAIRNESS:
- eMBB: Bandwidth = 90.0 / (embb_users + 1), then clamp to [6, 20] MHz
- URLLC: Bandwidth = 30.0 / (urllc_users + 1), then clamp to [1, 5] MHz

Example:
- eMBB slice has 4 users → Bandwidth = 90.0 / (4+1) = 18.0 MHz ✓
- URLLC slice has 9 users → Bandwidth = 30.0 / (9+1) = 3.0 MHz ✓

=== AVAILABLE TOOLS ===

**Tool 0: ray_tracing** [⚠️ CALL FIRST!]
📡 Purpose: Obtain accurate CQI from user position via ray tracing
- Use when: CQI is NOT provided in the problem (V3 dataset)
- Input: user_x (float), user_y (float), region (string, e.g. "HKUST_Center")
- Output: CQI value (1-15) based on path loss and SNR
- ⚠️ ALWAYS call this tool FIRST if CQI is not given!

**Tool 1: knowledge_base_query**
📚 Purpose: Query network slicing knowledge base for service classification
- Use when: Need to identify if service is eMBB or URLLC type
- Input: query (string) - describe the user request
- Output: Service type and recommended slice

**Tool 2: optimal_bandwidth_calculator** [MOST IMPORTANT]
🎯 Purpose: Calculate bandwidth using PROPORTIONAL FAIRNESS
- Use when: After determining slice type, calculate bandwidth
- Input: slice_type ("eMBB" or "URLLC"), existing_users (int)
- Output: Optimal bandwidth in MHz
- **This tool implements the proportional fairness formula!**

**Tool 3: shannon_calculator**
🔢 Purpose: Verify rate using Shannon formula
- Use when: Need to verify expected rate after bandwidth allocation
- Input: cqi (int), bandwidth (float)
- Output: Expected rate in Mbps
- Formula: Rate = 10.0 × Bandwidth × log₁₀(1 + 10^(CQI/10))

**Tool 4: slice_allocator**
📋 Purpose: Verify allocation meets constraints
- Use when: Final verification before output
- Input: slice_type, bandwidth, user_cqi
- Output: Allocation validation result

=== SLICE TYPE CLASSIFICATION ===

**eMBB (enhanced Mobile Broadband)** - Total: 90 MHz, Range: 6-20 MHz
- Video streaming, 4K/8K content
- AR/VR, VR gaming, augmented reality, navigation, maps
- File download, cloud storage, sync
- Web browsing, social media, music streaming, conference
⚠️ If service involves VR/AR/video + gaming, choose eMBB (bandwidth > latency)

**URLLC (Ultra-Reliable Low-Latency)** - Total: 30 MHz, Range: 1-5 MHz  
- Competitive gaming, esports (pure latency-sensitive)
- VoIP, voice calls
- Remote control, robotics, industrial automation
- IoT sensors, real-time monitoring, vital signs
- Mission-critical: surgery, safety, trading, emergency, early warning, fraud detection

=== DECISION WORKFLOW ===

Step 1: Get CQI via ray_tracing
- Extract user_x, user_y, and map region from the problem text
- Call ray_tracing(user_x, user_y, region) → returns CQI (1-15)
- ⚠️ This is MANDATORY — do NOT skip or guess CQI!

Step 2: Identify Service Type
- Extract keywords from user request
- If VR/AR/video/streaming → eMBB (even if "gaming" appears)
- If pure gaming/control/IoT/surgery/emergency → URLLC

Step 3: Extract Network State from Problem
- Count existing users in the target slice
- This is CRITICAL for proportional fairness calculation!

Step 4: Calculate Bandwidth (PROPORTIONAL FAIRNESS)
- Use optimal_bandwidth_calculator with slice_type and existing_users
- Example: eMBB with 4 users → 90/(4+1) = 18.0 MHz

Step 5: Output Final Answer
- Your FINAL ANSWER must contain ALL 4 fields in this exact format:

=== OUTPUT FORMAT ===

CQI: [value from ray_tracing, 1-15]
Slice Type: [eMBB or URLLC]
Bandwidth: [value] MHz
Throughput: [value] Mbps

Example:
CQI: 12
Slice Type: eMBB
Bandwidth: 18.00 MHz
Throughput: 220.91 Mbps

⚠️ ALL 4 FIELDS ARE REQUIRED! Missing any field = 0 score for that metric.
"""


# ============================================================================
# Python Code Solver Prompt
# ============================================================================

PYTHON_CODE_SOLVER_PROMPT = """
You are a Python programmer for network slicing calculations.

Problem: {problem}
Context: {context}

REQUIREMENTS:
1. Write a `solve()` function that returns the allocation decision
2. Use math library for logarithm calculations
3. Include the Shannon formula: Rate = 10.0 × B × log₁₀(1 + 10^(CQI/10))
4. Return a dictionary with: slice_type, allocated_bandwidth, expected_rate

Example structure:
```python
import math

def solve():
    # Given parameters
    cqi = {cqi}
    slice_type = "{slice_type}"
    
    # Shannon formula
    snr_linear = 10 ** (cqi / 10)
    spectral_efficiency = math.log10(1 + snr_linear)
    
    # Calculate bandwidth for target rate
    # ... your calculation ...
    
    return {{
        "slice_type": slice_type,
        "allocated_bandwidth": bandwidth,
        "expected_rate": rate
    }}
```
"""

# ============================================================================
# Verification Prompt
# ============================================================================

VERIFY_ALLOCATION_PROMPT = """
Verify the network slicing allocation:

User Request: {request}
Proposed Allocation:
- Slice Type: {slice_type}
- Bandwidth: {bandwidth} MHz
- Expected Rate: {rate} Mbps

Check:
1. Is the slice type appropriate for the service?
2. Is bandwidth within the slice constraints ({bw_min}-{bw_max} MHz)?
3. Does the rate satisfy minimum requirements?

Output: Verified allocation or corrected values.
"""
