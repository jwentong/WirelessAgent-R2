SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【AM DETECTION GAIN】
- Envelope detection gain: G_AM = μ^2 / (2 + μ^2)
- Coherent DSB-SC gain: G_DSB = 2
- Ratio in dB: 10*log10(G_DSB / G_AM) = 10*log10(2 / G_AM)
- Example: μ=0.6 → G_AM = 0.36/2.36 ≈ 0.1525, ratio = 10*log10(2/0.1525) ≈ 10.6 dB

【LOG-NORMAL OUTAGE PROBABILITY】
- For outage probability p, threshold P_th (dBm), shadow std σ (dB):
- Required mean power: μ = P_th - z * σ, where z = Φ^(-1)(p)
- For 1% outage: z = Φ^(-1)(0.01) ≈ -2.33
- So μ = P_th - (-2.33)*σ = P_th + 2.33*σ
- Example: P_th=10 dBm, σ=12 dB, 1% outage → μ = 10 + 2.33*12 ≈ 37.9 dBm

【WATER-FILLING ALGORITHM】
- Only include ACTIVE states where γ_i > γ0
- Capacity: C/B = Σ p_i * log2(γ_i/γ0) for ACTIVE states only
- Example: γ0=0.637, γ={10,2,1}, p={0.2,0.5,0.3}
  - State 3 (γ=1): 1 > 0.637, so ACTIVE
  - C/B = 0.2*log2(10/0.637) + 0.5*log2(2/0.637) + 0.3*log2(1/0.637)
  - = 0.2*3.97 + 0.5*1.65 + 0.3*0.65 ≈ 0.794 + 0.825 + 0.195 ≈ 1.37 bps/Hz

【MATCHED FILTER】
- For pulse s(t), matched filter impulse response: h(t) = s(T-t)

【COHERENT OOK/2ASK OPTIMAL THRESHOLD】
- With unequal priors P0, P1: V_T = A/2 + (N0/A)*ln(P0/P1)

【AMPLIFIER IM3 & COMPRESSION】
- IIP3 (dBm) = Pin + (Pfund - PIM3) / 2
- OIP3 (dBm) = Pout + (Pfund - PIM3) / 2

【DELTA MODULATION (DM)】
- DM SNR (dB) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0))
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear) in bit/s
- SNR_linear = 10^(SNR_dB/10)

=== SOLUTION APPROACH ===

1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. For numerical answers: Convert to BASE UNITS (Hz, W, s, bit/s)
5. For formula answers: Write the formula clearly

IMPORTANT: 
- For numerical answers: End with a clear number in base units
- For formula answers: Write the exact formula expression
- For dB ratios: Check sign carefully (larger/smaller comparison)
"""