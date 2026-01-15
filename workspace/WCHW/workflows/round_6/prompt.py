SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【DIGITAL MODULATION BER - VERY IMPORTANT】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0_linear))
  * First convert dB to linear: Eb/N0_linear = 10^(Eb/N0_dB / 10)
  * Example: Eb/N0 = 2 dB → linear = 10^(2/10) = 1.585
  * Then: BER = 0.5 * erfc(√1.585) = 0.5 * erfc(1.259) ≈ 0.0375
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【OOK DETECTION THRESHOLD】
- For coherent OOK with means 0 and A: optimal threshold V_T = A/2
- If A is given as a number, compute the actual value (e.g., A=4 → V_T=2)

【AM MODULATION GAIN】
- G_AM = μ^2/(2+μ^2) for envelope detection
- DSB-SC coherent gain = 2
- Ratio in dB: 10*log10(DSB-SC_gain / G_AM) = 10*log10(2 / G_AM)
- Note: This ratio is POSITIVE since DSB-SC gain > G_AM

【ANTENNA EFFICIENCY】
- Radiation efficiency η = R_rad / (R_rad + R_loss)
- If asked for multiple antennas, output the SMALLER efficiency value

【Q-FUNCTION AND ERFC RELATIONSHIP】
- Q(x) = 0.5 * erfc(x / √2)
- erfc(x) = 2 * Q(x * √2)

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- SNR_linear = 10^(SNR_dB/10)

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. ALWAYS convert dB to linear first: linear = 10^(dB/10)
3. Select the correct formula
4. Perform step-by-step calculations
5. For dB ratios: ensure correct order (reference/measured or better/worse)
6. Convert to base units: Hz (not kHz), W (not mW), s (not ms)
7. If multiple values computed, output the one specifically asked for

IMPORTANT: End your solution with a clear final answer as a single number.
"""