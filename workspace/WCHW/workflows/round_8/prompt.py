SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

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
- Coherent 2ASK/OOK: BER = Q(√(Eb/N0)), where Eb = A²/(4*Rs), N0 = n0/2

【AMPLIFIER NONLINEARITY】
- IIP3 (dBm) = Pin + (Pout_desired - Pout_IM3) / 2
- OIP3 (dBm) = Pout_desired + (Pout_desired - Pout_IM3) / 2

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【BANDWIDTH】
- NRZ first-null: B = Rb
- Raised-cosine: B = Rs(1+α)/2, where Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear) in bit/s
- SNR_linear = 10^(SNR_dB/10)
- IMPORTANT: Output in bit/s, NOT Mbit/s!

【NRZ SIGNALING BER】
- Bipolar NRZ: Pb = Q(√(2S/N)), so S/N = (Q^(-1)(Pb))^2 / 2
- Unipolar NRZ: Pb = Q(√(S/N)), so S/N = (Q^(-1)(Pb))^2

【LOG-NORMAL SHADOWING】
- Outage probability: P_out = Φ((P_req - μ)/σ)
- For P_out < 0.01: μ > P_req + 2.326*σ

【ANTENNA DIRECTIVITY】
- D = 4π / ∫G(θ)sin(θ)dθ
- 3-dB beamwidth: where G = 0.5*G_max
- 10-dB beamwidth: where G = 0.1*G_max

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. Convert to BASE UNITS: Hz (not kHz/MHz), W (not mW), s (not ms), bit/s (not Mbit/s)
5. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number in base units.
"""