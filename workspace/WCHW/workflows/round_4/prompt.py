SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== UNIT CONVERSION (CRITICAL - APPLY BEFORE FINAL ANSWER) ===
Your final answer MUST be in BASE SI UNITS with conversion already applied:
| Given Unit | Base Unit | Multiply by |
|------------|-----------|-------------|
| μs         | s         | 1e-6        |
| ms         | s         | 1e-3        |
| kHz        | Hz        | 1e3         |
| MHz        | Hz        | 1e6         |
| mW         | W         | 1e-3        |
| kbit/s     | bit/s     | 1e3         |

Example: 3.90625 μs → 3.90625 × 1e-6 = 3.90625e-06 (output this!)

=== CRITICAL FORMULAS ===

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0_linear))
  * Convert dB to linear: Eb/N0_linear = 10^(Eb/N0_dB / 10)
  * Example: Eb/N0 = 12 dB → linear = 10^(12/10) = 15.85
  * BER = 0.5 * erfc(√15.85) = 0.5 * erfc(3.98) ≈ 9e-9
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【COHERENT OOK THRESHOLD】
- Optimal threshold V_T = √(E_s)/2 for equal priors
- In terms of E_s coefficient: threshold = √E_s / 2, so coefficient is 0.5
- But if asking "threshold in terms of E_s" as a multiplier: answer is 2 (since V_T² = E_s/4)

【Q-FUNCTION AND ERFC】
- Q(x) = 0.5 * erfc(x / √2)
- erfc(x) = 2 * Q(x * √2)

【DELTA MODULATION (DM)】
- DM SNR (dB) = -13.6 + 30*log10(fs/fm)
- DO NOT use PCM formula for DM!

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)

=== SOLUTION APPROACH ===
1. Identify given values and units
2. Convert dB to linear if needed: linear = 10^(dB/10)
3. Apply correct formula
4. Calculate step by step
5. CONVERT FINAL ANSWER TO BASE SI UNITS
6. Output the numerical value only

IMPORTANT: Your final answer must be a NUMBER in base SI units (s, Hz, W, bit/s).
If you calculate 3.90625 μs, output 3.90625e-06
"""