# WCHW Round 1 - Enhanced Baseline Prompt with Formula Library
# This provides the foundation for MCTS to build upon

SOLVE_WITH_FORMULAS_PROMPT = """You are a telecommunications expert. Solve this problem step by step with careful attention to formulas and calculations.

=== CRITICAL FORMULAS (LLM often gets these wrong) ===

【DELTA MODULATION (DM) - NOT the same as PCM!】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3)
  where fs = sampling frequency, fm = max signal frequency
- DO NOT use PCM formula (6.02n + 1.76) for DM!
- DM step size for no slope overload: Δ ≥ 2πfm*A_max/fs

【QUANTIZATION & PCM】
- SQNR (dB) = 6.02n + 1.76 (for n-bit uniform quantization of sinusoid)
- Number of quantization levels: L = 2^n
- Step size: Δ = (V_max - V_min) / L

【COHERENT DIGITAL MODULATION BER】
- Coherent BPSK:     BER = 0.5 * erfc(√(Eb/N0))
- Coherent QPSK:     BER = 0.5 * erfc(√(Eb/N0))  [same as BPSK per bit]
- Coherent BFSK:     BER = 0.5 * erfc(√(Eb/(2*N0)))  [Note: 2 in denominator!]
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK:              BER = 0.5 * exp(-Eb/N0)

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm) where Δf = frequency deviation
- Modulation index: β = Δf/fm
- FM power: Carrier power stays constant (amplitude modulation changes)
- When amplitude doubles, Δf doubles (frequency deviation is proportional to amplitude)

【RAYLEIGH FADING CHANNEL】
- Level Crossing Rate: N_R = √(2π) * fD * ρ * exp(-ρ²)
  where ρ = threshold/RMS_level, fD = Doppler frequency
- Average Fade Duration: τ = (exp(ρ²) - 1) / (ρ * fD * √(2π))
- Markov model transition: Depends on correlation ρ = J0(2π*fD*Ts)

【BANDWIDTH FORMULAS】
- NRZ first-null: B = Rb (where Rb = bit rate)
- Raised-cosine: B = Rs(1+α)/2 where Rs = symbol rate, α = roll-off
- For M-ary modulation: Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- C = B * log2(1 + P/(N0*B))
- Convert dB to linear: SNR_linear = 10^(SNR_dB/10)

【A-LAW / μ-LAW COMPANDING】
- 8-bit codeword structure: [sign(1)][segment(3)][level(4)]
- E1 frame: 32 timeslots × 8 bits = 256 bits, frame duration = 125 μs

=== SOLUTION APPROACH ===

1. **Identify given values and what is asked**
2. **Select the appropriate formula** from above list (or derive if needed)
3. **Perform calculations step by step** - show all intermediate values
4. **For complex functions (erfc, log, exp)**: Set up the expression clearly
5. **State final answer as a clear number** with appropriate units

Important: 
- For context-dependent problems referencing "previous problem", infer from context
- For special function calculations (erfc, log2), verify numerically if possible
- Always end with the final numerical answer

Provide the final answer as a clear number.
"""