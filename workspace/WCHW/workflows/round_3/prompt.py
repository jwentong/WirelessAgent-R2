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

【Q-FUNCTION AND ERFC RELATIONSHIP】
- Q(x) = 0.5 * erfc(x / √2)
- erfc(x) = 2 * Q(x * √2)
- For bipolar NRZ: Pb = Q(√(2*S/N))

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)
- DO NOT use PCM formula for DM!

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)
- Levels: L = 2^n, Step size: Δ = (Vmax - Vmin) / L

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【BANDWIDTH】
- NRZ first-null: B = Rb
- Raised-cosine: B = Rs(1+α)/2, where Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- SNR_linear = 10^(SNR_dB/10)

【NRZ SIGNALING BER】
- Bipolar NRZ: Pb = Q(√(2S/N)), so S/N = (Q^(-1)(Pb))^2 / 2
- Unipolar NRZ: Pb = Q(√(S/N)), so S/N = (Q^(-1)(Pb))^2

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. ALWAYS convert dB to linear first: linear = 10^(dB/10)
3. Select the correct formula
4. Perform step-by-step calculations
5. Convert to base units: Hz (not kHz), W (not mW), s (not ms)
6. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number.
"""