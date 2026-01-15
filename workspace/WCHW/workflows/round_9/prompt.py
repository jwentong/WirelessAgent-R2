SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【SHANNON CAPACITY - VERY IMPORTANT】
- C = B * log2(1 + SNR_linear) where B is in Hz
- For multi-user: C1 = B * log2(1 + P1/(P2+N))
- Use natural log: log2(x) = ln(x) / ln(2) = ln(x) / 0.693147
- Example: B=5 MHz, P1=4mW, P2=1mW, N=1mW
  * C1 = 5e6 * log2(1 + 4/(1+1)) = 5e6 * log2(3) = 5e6 * 1.585 = 7.92e6 bps
  * BUT if formula is C1 = B*log2(1+P1/(P2+N)), compute exactly!

【Q-FUNCTION CALCULATIONS】
- Q(x) = 0.5 * erfc(x / √2)
- For Pb = Q(√(2*S/N)) with S/N = 20 (linear):
  * √(2*20) = √40 = 6.324
  * Q(6.324) ≈ 1.27e-10 to 2.1e-10
- Q(6) ≈ 9.87e-10, Q(6.5) ≈ 4.02e-11
- Be very precise with Q-function values!

【SPECTRAL EFFICIENCY】
- η = Rb / B (bits per second per Hz)
- For binary NRZ with 2PSK: symbol rate Rs = B (Nyquist), Rb = Rs * 1 bit
- BUT bandwidth of NRZ = 2*Rs, so η = Rs/(2*Rs) = 0.5 bit/s/Hz
- For M-ary: η = log2(M) * Rs / B

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0_linear))
  * First convert dB to linear: Eb/N0_linear = 10^(Eb/N0_dB / 10)
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【OOK DETECTION THRESHOLD】
- For coherent OOK with means 0 and A: optimal threshold V_T = A/2

【AM MODULATION GAIN】
- G_AM = μ^2/(2+μ^2) for envelope detection
- DSB-SC coherent gain = 2

【ANTENNA EFFICIENCY】
- Radiation efficiency η = R_rad / (R_rad + R_loss)

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. ALWAYS convert dB to linear first: linear = 10^(dB/10)
3. Select the correct formula
4. Perform step-by-step calculations with high precision
5. Convert to base units: Hz (not kHz), W (not mW), s (not ms), bit/s (not kbit/s)
6. For spectral efficiency, check if NRZ bandwidth = 2*symbol_rate

IMPORTANT: End your solution with a clear final answer as a single number.
"""