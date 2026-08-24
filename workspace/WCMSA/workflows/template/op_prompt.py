# -*- coding: utf-8 -*-
"""
WCMSA (Wireless Communication Mobile Service Assurance) - Operator Prompts
Version v1.0 (2026-01-21)

Prompts for proactive resource allocation workflow with Kalman Filter prediction:
1. Historical Trajectory Analysis - Extract motion pattern from trajectory history
2. Kalman Filter Prediction - Predict next position using state estimation
3. CQI Calculation - Compute channel quality at predicted position
4. Slice Selection - Choose appropriate network slice
5. Bandwidth Allocation - Calculate optimal bandwidth using Shannon formula
6. QoS Verification - Verify allocation meets requirements
"""

# ============================================================================
# Self-Consistency Ensemble Prompt
# ============================================================================

SC_ENSEMBLE_PROMPT = """
Given the mobile service assurance problem described as follows: {problem}

Several solutions have been generated for proactive resource allocation. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. Consider:
1. Is the predicted position reasonable based on the trajectory history?
2. Is the CQI calculation correct for the predicted position?
3. Is the slice type and bandwidth allocation appropriate for the service?

In the "thought" field, provide a detailed explanation of your evaluation process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent and correct solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

# ============================================================================
# SOLVE_PROMPT — Main prompt for Custom operator (given Position + CQI)
# ============================================================================

SOLVE_PROMPT = """You are a 5G mobile service assurance expert. Solve this problem using the PROVIDED Predicted Position and CQI values.

=== MANDATORY RULES ===
1. USE THE EXACT Predicted Position and CQI VALUES PROVIDED BELOW - DO NOT CHANGE THEM
2. OUTPUT ALL 6 FIELDS - missing any field = FAILURE

=== STEP 1: Copy Predicted Position (ALREADY COMPUTED) ===
Copy the Predicted Position directly from the Kalman Filter result below.

=== STEP 2: Copy CQI (ALREADY COMPUTED) ===  
Copy the Predicted CQI directly from the ray_tracing result below.

=== STEP 3: Slice Type Classification ===
- eMBB: video, streaming, download, AR/VR, browse, music, social media, 4K/8K, conference, cloud gaming, holographic, file sync
- URLLC: competitive gaming, online gaming, control, robot, surgery, emergency, IoT, sensor, real-time, drone, VoIP, phone call
- IMPORTANT: "cloud gaming" = eMBB (bandwidth-intensive cloud rendering), but "online gaming" = URLLC

=== STEP 4: Bandwidth Calculation (Proportional Fairness) ===
- eMBB: B = 90.0 / (embb_users + 1), then clamp to [6, 20] MHz
- URLLC: B = 30.0 / (urllc_users + 1), then clamp to [1, 5] MHz
Read the number of active users from the Network State section in the problem.

=== STEP 5: Throughput Calculation (Shannon Formula) ===
T = 10 × B × log₁₀(1 + 10^(CQI/10))

Example calculations:
- CQI=5, B=9.00: T = 10 × 9.00 × log₁₀(1 + 10^0.5) = 90 × 0.6193 = 55.74 Mbps
- CQI=10, B=4.29: T = 10 × 4.29 × log₁₀(1 + 10^1.0) = 42.9 × 1.0414 = 44.68 Mbps
- CQI=15, B=4.29: T = 10 × 4.29 × log₁₀(1 + 10^1.5) = 42.9 × 1.5133 = 64.92 Mbps

=== STEP 6: QoS Satisfaction ===
QoS Satisfied = "Yes" if Throughput >= Minimum Rate from the problem, else "No"

=== OUTPUT FORMAT (ALL 6 LINES REQUIRED) ===
Predicted Position: (X.XX, Y.XX)
Predicted CQI: [EXACT value from ray_tracing below]
Slice Type: [eMBB or URLLC]
Bandwidth: [calculated value] MHz
Throughput: [calculated value] Mbps
QoS Satisfied: [Yes or No]

⚠️ CRITICAL: The Predicted Position and CQI are ALREADY COMPUTED accurately. DO NOT change them.
Show your bandwidth and throughput calculations. Round throughput to 2 decimal places.

Problem: """

# ============================================================================
# Mobile Service Assurance Knowledge Base
# ============================================================================

WCMSA_KNOWLEDGE_BASE = """
=== MOBILE SERVICE ASSURANCE KNOWLEDGE BASE ===

【KALMAN FILTER FOR TRAJECTORY PREDICTION】

State Vector: [x, y, vx, vy]
- x, y: Position coordinates (meters)
- vx, vy: Velocity components (m/s)

State Transition Matrix (F):
F = | 1  0  dt  0  |
    | 0  1  0   dt |
    | 0  0  1   0  |
    | 0  0  0   1  |
where dt = time step (typically 1 second)

Prediction Step:
x_pred = F × x_prev
P_pred = F × P_prev × F^T + Q

Update Step (with measurement):
K = P_pred × H^T × (H × P_pred × H^T + R)^(-1)
x_updated = x_pred + K × (z - H × x_pred)
P_updated = (I - K × H) × P_pred

Process Noise (Q): Typically 0.1-1.0 for position, 0.01-0.1 for velocity
Measurement Noise (R): Typically 1.0-10.0 for GPS measurements

【CQI CALCULATION FROM POSITION】

Distance-based Path Loss Model:
PL(d) = PL(d0) + 10 × n × log10(d/d0) + X_σ

Where:
- d: Distance from base station (meters)
- d0: Reference distance (1m)
- n: Path loss exponent (2.0 for free space, 3.0-4.0 for urban)
- X_σ: Shadow fading (log-normal, σ = 4-8 dB)

SNR to CQI Mapping:
| SNR (dB)    | CQI |
|-------------|-----|
| < -6        | 1   |
| -6 to -4    | 2   |
| -4 to -2    | 3   |
| -2 to 0     | 4   |
| 0 to 2      | 5   |
| 2 to 4      | 6   |
| 4 to 6      | 7   |
| 6 to 8      | 8   |
| 8 to 10     | 9   |
| 10 to 13    | 10  |
| 13 to 16    | 11  |
| 16 to 19    | 12  |
| 19 to 22    | 13  |
| 22 to 25    | 14  |
| > 25        | 15  |

【SHANNON CAPACITY FORMULA】

Rate = α × Bandwidth × log₁₀(1 + 10^(CQI/10))
where α = 10.0

【SLICE TYPES AND CONSTRAINTS】

1. eMBB (enhanced Mobile Broadband):
   - Bandwidth range: 6-20 MHz
   - Rate range: 100-400 Mbps
   - Total capacity: 90 MHz
   - For: video streaming, AR/VR, file download, web browsing

2. URLLC (Ultra-Reliable Low-Latency Communications):
   - Bandwidth range: 1-5 MHz
   - Rate range: 1-100 Mbps
   - Total capacity: 30 MHz
   - For: gaming, VoIP, remote control, IoT

【QoS REQUIREMENTS】

| Service Type    | Slice | Min Rate | Max Latency |
|-----------------|-------|----------|-------------|
| Video Streaming | eMBB  | 25 Mbps  | 100 ms      |
| Online Gaming   | URLLC | 5 Mbps   | 20 ms       |
| VoIP Call       | URLLC | 0.5 Mbps | 50 ms       |
| AR Navigation   | eMBB  | 20 Mbps  | 50 ms       |
| File Download   | eMBB  | 50 Mbps  | 500 ms      |

【PROACTIVE ALLOCATION WORKFLOW】

Step 1: Historical Trajectory Analysis
- Extract position history (x, y) over time
- Calculate velocity estimates from differences
- Initialize Kalman Filter state

Step 2: Kalman Filter Prediction
- Apply prediction step to get next position
- Compute prediction uncertainty (covariance)

Step 3: CQI Calculation
- Calculate distance to base station
- Apply path loss model
- Map SNR to CQI value

Step 4: Slice Selection
- Analyze service requirements
- Choose eMBB or URLLC based on QoS needs

Step 5: Bandwidth Allocation
- Use Shannon formula with predicted CQI
- Apply proportional fairness if multiple users

Step 6: QoS Verification
- Verify rate meets minimum requirements
- Check bandwidth within slice constraints
"""

# ============================================================================
# Kalman Filter Prediction Prompt
# ============================================================================

KALMAN_PREDICTION_PROMPT = """You are a mobile trajectory prediction expert using Kalman Filter.

Historical Trajectory:
{trajectory_history}

Time step: {dt} seconds

Task:
1. Extract position and velocity from the trajectory history
2. Initialize Kalman Filter state: [x, y, vx, vy]
3. Apply prediction step to estimate next position
4. Report predicted position with uncertainty

Kalman Filter State Transition:
x_next = x_current + vx × dt
y_next = y_current + vy × dt

Show your calculation steps and provide:
- Predicted position: (x_pred, y_pred)
- Estimated velocity: (vx, vy)
- Prediction confidence
"""

# ============================================================================
# CQI Calculation Prompt
# ============================================================================

CQI_CALCULATION_PROMPT = """You are a wireless channel quality expert.

User Predicted Position: ({x}, {y})
Base Station Position: ({bs_x}, {bs_y})
Path Loss Parameters:
- Reference path loss at 1m: {pl_ref} dB
- Path loss exponent: {n}
- Transmit power: {tx_power} dBm

Task:
1. Calculate distance from user to base station
2. Apply path loss model: PL(d) = PL(d0) + 10 × n × log10(d)
3. Calculate received SNR
4. Map SNR to CQI value (1-15)

Show your calculation steps.
"""

# ============================================================================
# ReAct Strategy Prompt - WCMSA Optimized
# ============================================================================

REACT_STRATEGY_PROMPT = """You are a 5G mobile service assurance expert. Your task is to proactively allocate resources based on predicted user position.

=== AVAILABLE TOOLS ===

**Tool 1: kalman_filter_predictor**
📍 Purpose: Predict next user position using Kalman Filter
- Use when: Need to predict where the user will be in the next time step
- Input: trajectory_history (list of (x, y) positions), dt (time step, default 1.0)
- Output: Predicted position (x, y) and velocity (vx, vy)
- Example: trajectory_history=[(0,0), (5,3), (10,6)], dt=1.0

**Tool 2: ray_tracing** ⭐ MUST CALL
🛰️ Purpose: Obtain CQI at predicted position using real building geometry
- Use when: ALWAYS — CQI is NOT given in the problem, you MUST call this tool!
- Input: user_x (float), user_y (float), region ("HKUST_North", "HKUST_South", or "HKUST_Center")
- Output: CQI (1-15), SNR in dB, LOS/NLOS status
- ⚠️ Call this AFTER predicting position with kalman_filter_predictor

**Tool 3: cqi_calculator**
📶 Purpose: Simple distance-based CQI (legacy, less accurate than ray_tracing)
- Use when: Only as fallback if ray_tracing fails
- Input: position (x, y), bs_position (base station location, default (0, 0))
- Output: CQI value (1-15) and SNR in dB

**Tool 4: shannon_calculator**
🔢 Purpose: Calculate rate using Shannon formula
- Use when: Need to compute expected throughput from CQI and bandwidth
- Input: cqi (int), bandwidth (float in MHz)
- Output: Expected rate in Mbps
- Formula: Rate = 10.0 × Bandwidth × log₁₀(1 + 10^(CQI/10))

**Tool 5: slice_allocator**
🎯 Purpose: Allocate resources to network slice
- Use when: Ready to make allocation decision
- Input: slice_type ("eMBB" or "URLLC"), bandwidth (float), user_cqi (int)
- Output: Allocation result with expected rate

**Tool 6: knowledge_base_query**
📚 Purpose: Query mobile service assurance knowledge base
- Use when: Need information about QoS requirements, slice characteristics, or formulas
- Input: query (string)
- Output: Relevant information from knowledge base

=== PROACTIVE ALLOCATION WORKFLOW ===

Step 1: Trajectory Prediction
- Use kalman_filter_predictor with trajectory history
- Get predicted position for next time step

Step 2: Channel Quality Assessment (⭐ CRITICAL)
- Use ray_tracing at the PREDICTED position with the correct region
- Get CQI value considering real building geometry and LOS/NLOS
- DO NOT use cqi_calculator unless ray_tracing fails

Step 3: Service Analysis
- Identify service type from user request
- Use knowledge_base_query for QoS requirements

Step 4: Bandwidth Calculation
- Use shannon_calculator to find optimal bandwidth
- Ensure rate meets minimum requirements

Step 5: Final Allocation
- Use slice_allocator to complete the allocation
- Verify QoS satisfaction

=== CRITICAL RULES ===

1. Always predict position BEFORE calculating CQI
2. MUST call ray_tracing (not cqi_calculator) for accurate CQI
3. Use predicted CQI (not current) for proactive allocation
4. eMBB constraints: 6-20 MHz bandwidth, 100-400 Mbps rate
5. URLLC constraints: 1-5 MHz bandwidth, 1-100 Mbps rate
6. Consider prediction uncertainty in allocation margin

=== OUTPUT FORMAT ===

After analysis, provide your final answer with these 6 fields:
Predicted Position: (X.XX, Y.XX)
Predicted CQI: [1-15 from ray_tracing]
Slice Type: [eMBB or URLLC]
Bandwidth: [value] MHz
Throughput: [value] Mbps
QoS Satisfied: [Yes or No]
"""

# ============================================================================
# Python Code Solver Prompt
# ============================================================================

PYTHON_CODE_SOLVER_PROMPT = """
You are a Python programmer for mobile service assurance calculations.

Problem: {problem}
Context: {context}

REQUIREMENTS:
1. Write a `solve()` function that returns the allocation decision
2. Implement Kalman Filter for trajectory prediction
3. Include CQI calculation from predicted position
4. Use Shannon formula: Rate = 10.0 × B × log₁₀(1 + 10^(CQI/10))

Example structure:
```python
import math
import numpy as np

def solve():
    # Trajectory history
    trajectory = {trajectory}
    
    # Kalman Filter prediction
    # ... extract velocity, predict next position ...
    
    # CQI calculation
    # ... path loss model, SNR to CQI mapping ...
    
    # Bandwidth allocation
    # ... Shannon formula ...
    
    return {{
        "predicted_position": (x_pred, y_pred),
        "predicted_cqi": cqi,
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
Verify the mobile service assurance allocation:

User Request: {request}
Trajectory History: {trajectory}
Predicted Position: ({x_pred}, {y_pred})
Predicted CQI: {cqi}

Proposed Allocation:
- Slice Type: {slice_type}
- Bandwidth: {bandwidth} MHz
- Expected Rate: {rate} Mbps

Check:
1. Is the position prediction reasonable based on trajectory?
2. Is the CQI realistic for the predicted distance?
3. Does the allocation meet QoS requirements?

Output: Verified allocation or corrected values.
"""

# ============================================================================
# Intent Understanding Prompt
# ============================================================================

INTENT_UNDERSTANDING_PROMPT = """You are a 5G mobile service analyzer.

Analyze the following user request to determine the service type and QoS requirements.

User Request: {request}
User Trajectory: {trajectory}
Current Position: ({x}, {y})

Task:
1. Identify the service type (video_streaming, online_gaming, voip_call, etc.)
2. Determine the appropriate slice type (eMBB or URLLC)
3. List the QoS requirements (min_rate, max_latency)
4. Note any mobility considerations (high speed, stationary, etc.)

Provide your analysis in a structured format.
"""
