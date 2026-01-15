SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【OUTAGE CAPACITY】
- C/B = Σ P(γ≥γ_min) * log2(1+γ) for all γ≥γ_min
- Only sum over states where γ≥γ_min (exclude outage states)
- Weight each log2(1+γ) by its probability

【NOMA / MULTI-USER UPLINK】
- User decoded first (treated as interference): C1 = B*log2(1 + P1/(P2+N))
- User decoded second (after SIC): C2 = B*log2(1 + P2/N)
- IMPORTANT: Use natural log2, result in bits/s

【SPECTRAL EFFICIENCY】
- η = Rb / B (bit/s per Hz)
- For binary NRZ with 2PSK: η = 1 if Rb = B

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)
- DO NOT use PCM formula for DM!

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【DIGITAL MODULATION BER - CRITICAL dB CONVERSION】
- FIRST: Convert Eb/N0 from dB to linear: (Eb/N0)_linear = 10^((Eb/N0)_dB/10)
- Example: 2 dB → 10^(2/10) = 10^0.2 ≈ 1.585
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0)_linear)
- For Eb/N0 = 2 dB: BER = 0.5 * erfc(√1.585) = 0.5 * erfc(1.259) ≈ 0.0375
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【2ASK ENERGY CALCULATION】
- Eb = A²/(4*Rs) where T = 1/Rs

【AVERAGE FADE DURATION】
- T̄ = (e^(ρ²) - 1) / (√(2π) * fD * ρ)

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)

【BANDWIDTH】
- NRZ first-null: B = Rb
- Raised-cosine: B = Rs(1+α)/2, where Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- SNR_linear = 10^(SNR_dB/10)

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Convert dB to linear if needed: linear = 10^(dB/10)
3. Select the correct formula
4. Perform step-by-step calculations
5. Convert to base units: Hz (not kHz), W (not mW), s (not ms)
6. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number.
"""