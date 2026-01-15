SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== UNIT CONVERSION FIRST (CRITICAL!) ===
BEFORE any calculation, convert ALL values to base SI units:
- mW → W: divide by 1000 (10 mW = 0.01 W)
- kHz → Hz: multiply by 1000
- MHz → Hz: multiply by 1,000,000
- ms → s: divide by 1000
- μs → s: divide by 1,000,000

=== PATH LOSS CALCULATION ===
Path Loss (dB) = 10 * log10(P_t / P_r)
- MUST convert P_t to Watts first!
- Example: P_t = 10 mW = 0.01 W, P_r = 10^-12 W
- Path Loss = 10 * log10(0.01 / 10^-12) = 10 * log10(10^10) = 100 dB
- Wait, check: 10 * log10(10^-2 / 10^-12) = 10 * log10(10^10) = 100 dB

=== CRITICAL FORMULAS ===

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0_linear))
- Convert dB to linear: Eb/N0_linear = 10^(Eb/N0_dB / 10)
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【WATER-FILLING】
- Cutoff γ0: solve 1/γ0 = 1/γ_i + λ for active channels
- Total power constraint: Σ p_i * (1/γ0 - 1/γ_i) = P_total

【DELTA MODULATION (DM)】
- DM SNR (dB) = -13.6 + 30*log10(fs/fm)
- DO NOT use PCM formula for DM!

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)

【SYNDROME DECODER】
- If syndrome is all zeros → no error (output: no error)

=== SOLUTION APPROACH ===
1. Convert ALL units to base SI units FIRST
2. Identify the correct formula
3. Perform step-by-step calculations
4. State the final answer clearly

For conceptual questions (like syndrome decoder), give the text answer.
For numerical questions, give the number in base units.
"""