SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【WATER-FILLING ALGORITHM】
- Cutoff equation: p_total/γ0 = Σ p_i * max(1/γ0 - 1/γ_i, 0) for active states
- For active states (γ_i > γ0): p_total/γ0 = Σ p_i/γ0 - Σ p_i/γ_i
- Rearranging: p_total = Σ p_i - γ0 * Σ(p_i/γ_i)
- Solve: γ0 = (Σ p_i - p_total) / Σ(p_i/γ_i)

【8PSK BER (NEAREST-NEIGHBOR UNION BOUND)】
- p_b ≈ (2/3) * Q(√(6*Eb/N0) * sin(π/8))
- sin(π/8) ≈ 0.3827
- Q(x) = 0.5 * erfc(x/√2)
- For Eb/N0 in dB: convert to linear first: (Eb/N0)_linear = 10^(dB/10)

【MODULATION ORDER CALCULATION】
- Given: Rb = 2B * log2(M) / (1+α)
- Solve for M: log2(M) = Rb * (1+α) / (2B)
- M = 2^(Rb*(1+α)/(2B))
- IMPORTANT: Round UP to next power of 2 (e.g., if M=5.6, answer is 8)

【SPECTRAL EFFICIENCY】
- η = Rb / B (bit/s per Hz)
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

【AVERAGE FADE DURATION】
- T̄ = (e^(ρ²) - 1) / (√(2π) * fD * ρ)

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