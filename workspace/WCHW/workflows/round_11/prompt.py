SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【SPECTRAL EFFICIENCY】
- η = Rb / B (bit/s per Hz)
- For binary NRZ with 2PSK: If channel bandwidth B is given and NRZ first-null BW = Rb, then Rb = B, so η = 1
- BUT if B is the allocated channel and Rb = B/2 (Nyquist), then η = 0.5
- For M-ary: η = log2(M) * Rs / B

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)
- DO NOT use PCM formula for DM!

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)
- Levels: L = 2^n, Step size: Δ = (Vmax - Vmin) / L

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0))
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)
- Coherent 2ASK: BER = Q(√(Eb/N0)), where Eb = A²/(4*Rs)

【2ASK ENERGY CALCULATION】
- For 2ASK with amplitude A and symbol rate Rs:
- Eb = A²*T/4 = A²/(4*Rs) where T = 1/Rs
- If double-sided PSD is n0, then N0 = n0/2

【AVERAGE FADE DURATION】
- T̄ = (e^(ρ²) - 1) / (√(2π) * fD * ρ)
- Compute e^(ρ²) carefully, then subtract 1

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【BANDWIDTH】
- NRZ first-null: B = Rb
- Raised-cosine: B = Rs(1+α)/2, where Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- SNR_linear = 10^(SNR_dB/10)

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. Convert to base units: Hz (not kHz), W (not mW), s (not ms)
5. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number.
"""