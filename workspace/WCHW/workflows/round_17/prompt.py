SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【PATH LOSS】
- Path Loss (dB) = P_tx (dBm) - P_rx (dBm)
- Convert all powers to same unit first: 1 W = 30 dBm, 1 mW = 0 dBm
- Example: P_tx=1W=30dBm, P_rx=-80dBm → Loss = 30-(-80) = 110 dB? NO!
- Check if P_tx is given in dBm already. If P_tx=-30dBm, P_rx=-80dBm → Loss = 50 dB

【WATER-FILLING CAPACITY】
- C/B = Σ p_i * log2(γ_i/γ0) for ONLY states where γ_i > γ0
- If γ_i ≤ γ0, that channel is OFF (do not include in sum)

【SPECTRAL EFFICIENCY】
- η = Rb / B (bit/s per Hz)

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0))
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【SNR TO POWER】
- P_s = SNR_linear × N
- SNR_linear = 10^(SNR_dB/10)
- If SNR=15dB, SNR_linear = 10^1.5 ≈ 31.62

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. Convert to base units: Hz (not kHz), W (not mW), s (not ms)
5. DOUBLE-CHECK: Is your answer in the right order of magnitude?
6. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number.
"""