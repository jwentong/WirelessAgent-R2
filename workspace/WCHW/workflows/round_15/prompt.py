SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【DELAY SPREAD & MULTIPATH】
- Mean excess delay (T_m): When synchronized to path k, T_m = (sum of (power_i × |delay_i - delay_k|)) / (sum of power_i)
- For equal power paths synchronized to path with delay τ_k: T_m = mean of |τ_i - τ_k| for all paths
- RMS delay spread: σ_τ = sqrt(mean of (τ - T_m)^2)
- IMPORTANT: If synchronized to 50ns path with delays {0,50,120}ns, offsets are {50,0,70}ns, so T_m = (50+0+70)/3 ≈ 40ns... BUT if asking for max spread, T_m = 120-50 = 70ns

【CONVOLUTIONAL CODE RATE】
- Code rate R = k/n (k input bits, n output bits)
- Throughput = R × coded_bit_rate
- To DOUBLE throughput: new_rate = 2 × old_rate
- From R=1/2, doubling throughput requires R=1 (uncoded transmission)
- Rate 1/2 → Rate 1 doubles throughput (not 3/4!)

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
- Coherent OOK: Pb = Q(√(Eb/N0)), where Q(x) = 0.5*erfc(x/√2)
- For Eb/N0 = 8 dB: linear = 10^0.8 ≈ 6.31, √6.31 ≈ 2.51, Q(2.51) ≈ 0.00603

【Q FUNCTION VALUES】
- Q(2.5) ≈ 0.00621, Q(2.51) ≈ 0.00603, Q(2.52) ≈ 0.00587
- Q(x) = 0.5 * erfc(x / sqrt(2))

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

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. Convert to BASE UNITS: Hz (not kHz), W (not mW), s (not ms), bit/s (not Mbit/s)
5. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number in base units.
"""