"""
WCHW (Wireless Communication Homework) Dataset - Operator Prompts
Version v4.0 (2025-11-30)

Note: Unit conversion is now handled by WCHWBenchmark
Focus prompts on problem-solving logic, not unit formatting
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

Show your work clearly:
1. Identify what is given and what is asked
2. Select the appropriate formula(s)
3. Perform calculations step by step
4. State your final numerical answer

Provide the final answer as a clear number.
"""

# ============================================================================
# EXTENDED FORMULA LIBRARY - Covers commonly missed formulas
# ============================================================================

FORMULA_LIBRARY = """
=== CRITICAL FORMULAS (LLM often gets these wrong) ===

【DELTA MODULATION (DM) - NOT the same as PCM!】
- DM SNR (dB) = -13.60 + 30*log10(fs/fm)
  where fs = sampling frequency, fm = max signal frequency
  (Note: Some textbooks use -4.26, but WCHW dataset uses -13.60)
- DO NOT use PCM formula (6.02n + 1.76) for DM!
- DM step size for no slope overload: Δ ≥ 2πfm*A_max/fs

【RAISED COSINE BANDWIDTH - CRITICAL: divide by 2!】
- First-null bandwidth: B = Rs*(1+α)/2
  where Rs = symbol rate, α = roll-off factor
- For M-ary modulation: Rs = Rb/log2(M)
- Example: 8-PSK at 9600 bps, α=0.25 → Rs=3200, B=3200*1.25/2=2000 Hz
- DO NOT use B = Rs*(1+α) - this gives 2x the correct answer!

【COHERENT DIGITAL MODULATION BER】
- Coherent BPSK:     BER = 0.5 * erfc(√(Eb/N0))
- Coherent QPSK:     BER = 0.5 * erfc(√(Eb/N0))  [same as BPSK per bit]
- Coherent BFSK:     BER = 0.5 * erfc(√(Eb/(2*N0)))  [Note: 2 in denominator!]
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK:              BER = 0.5 * exp(-Eb/N0)

【RAYLEIGH FADING CHANNEL】
- Level Crossing Rate: N_R = √(2π) * fD * ρ * exp(-ρ²)
  where ρ = threshold/RMS_level (normalized), fD = max Doppler frequency
- Average Fade Duration: T_fade = (exp(ρ²) - 1) / (ρ * fD * √(2π))
- For Markov model: Use transition probabilities, NOT uniform distribution

【NAKAGAMI-m FADING BER】
- DPSK over Nakagami-m: P_b = (1/2)(1-μ)^m * Σ C(m-1+k,k)((1+μ)/2)^k
  where μ = √(γ_b/(1+γ_b)), γ_b = average SNR per bit
- For m=1 (Rayleigh): simplifies to standard Rayleigh formulas

【FM MODULATION】
- Carson's Rule: BW ≈ 2(Δf + fm)
- Modulation index: β = Δf/fm
- FM SNR improvement: G_FM = (3/2)β²(β+1) for wideband FM
- If signal amplitude doubles: Δf doubles, β doubles, BW changes

【AM MODULATION】
- AM power: P_total = P_c(1 + m²/2) for single-tone
- AM SNR improvement: G_AM = m²/(2+m²)
- SSB power: P_SSB = P_c*m²/4

【A-LAW / μ-LAW COMPANDING】
- 8-bit codeword structure: [sign(1)][segment(3)][level(4)]
- A-law segment step sizes: Δₙ = 2^n × Δ₀ for segment n
- Reconstruction: value = sign × (base + level × step_size)

【BANDWIDTH FORMULAS】
- NRZ first-null: B = Rb (where Rb = bit rate)
- Raised-cosine: B = Rs*(1+α)/2 where Rs = symbol rate, α = roll-off
- Nyquist minimum: B = Rs/2
- For M-ary: Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- C = B * log2(1 + P/(N0*B))
- Convert dB to linear: SNR_linear = 10^(SNR_dB/10)

【PCM】
- Bit rate: Rb = fs × n (fs = sampling rate, n = bits/sample)
- SQNR (dB) = 6.02n + 1.76 (for sinusoidal input)
- Quantization levels: L = 2^n
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer for MATHEMATICAL PROBLEM SOLVING.

Problem: {problem}
Context: {analysis}
{feedback}

REQUIREMENTS:
1. Write a `solve()` function that returns the final numerical answer
2. Use appropriate libraries: math, numpy, scipy if needed
3. Add detailed comments explaining each step
4. Include print statements for debugging intermediate results

The output should be limited to basic data types such as strings, integers, and floats.
"""


# Simple verification prompt for answer checking
VERIFY_ANSWER_PROMPT = """
Given the question and the proposed answer, verify if the answer is correct.

Question: {problem}
Proposed Answer: {answer}

Check:
1. Are the calculations correct?
2. Does the answer make physical sense?

Output only the verified/corrected numerical answer.
"""

# ReAct strategy prompt - WCHW optimized v5.2 (Clear Tool Roles)
REACT_STRATEGY_PROMPT = """You are a telecommunications expert solving wireless communication problems.

=== TOOL ROLES (Choose ONE per problem) ===

**Option A: RETRIEVAL TOOL** (telecom_formula_retriever)
📚 Purpose: Improve problem UNDERSTANDING
- Use when: Problem involves unfamiliar concepts, formulas, or terminology
- Use when: Need to recall correct formula before solving
- Use when: Unsure which approach to take
- After retrieval: Use your reasoning to compute the answer
- Example: "What is the BER formula for coherent BFSK?"

**Option B: CODE TOOL** (python_code_solver)
💻 Purpose: VERIFY calculations or solve complex computations
- Use when: You already understand the problem and know the formula
- Use when: Need precise calculation (erfc, Bessel, Marcum Q, etc.)
- Use when: Want to verify your mental calculation
- Example: "Compute 0.5*erfc(sqrt(10)) precisely"

=== DECISION RULE ===
Ask yourself: "What do I need help with?"

🤔 "I don't understand the problem/formula" → telecom_formula_retriever
✅ "I understand, but need precise calculation" → python_code_solver

⚠️ Use only ONE tool per problem, then provide final_answer.

=== CRITICAL FORMULAS (for reference) ===

【BANDWIDTH】B = Rs*(1+α)/2, Rs = Rb/log2(M)
【BER BPSK】0.5*erfc(sqrt(Eb/N0))
【BER BFSK coherent】0.5*erfc(sqrt(Eb/(2*N0)))  # Factor of 2!
【DM SNR】SNR_dB = -13.60 + 30*log10(fs/fm)
【Shannon】C = B*log2(1 + SNR_linear)

=== WORKFLOW ===
1. Read problem → Identify what you need (understanding vs calculation)
2. Choose ONE tool based on your need
3. Use tool once → Get result
4. Provide final_answer

Remember: One tool call, then answer!
"""