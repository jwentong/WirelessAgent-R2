SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【Q-FUNCTION & ERROR PROBABILITY】
- Q(x) = 0.5 * erfc(x/√2)
- For Pb = Q(√(S/N)) with S/N = 20 (linear):
  - √20 ≈ 4.472
  - Q(4.472) = 0.5 * erfc(4.472/1.414) = 0.5 * erfc(3.162) ≈ 3.87e-6
- erfc(3) ≈ 2.2e-5, erfc(3.5) ≈ 7.4e-7, erfc(4) ≈ 1.5e-8

【RAYLEIGH CLIPPING MODEL】
- Clipping probability: P_clip = exp(-A0^2)
- To find A0: A0 = sqrt(-ln(P_clip))
- Example: P_clip = 0.005 → A0 = sqrt(-ln(0.005)) = sqrt(5.298) ≈ 2.302
- CAUTION: It's exp(-A0^2), NOT exp(-A0)!

【AMPLIFIER IIP3 & OIP3】
- From 2-tone test: IIP3 = Pin + (ΔdB)/2 where ΔdB = Pfund - PIM3
- OIP3 = Pout + (ΔdB)/2
- Relationship: OIP3 = IIP3 + Gain
- Example: Pin=-6dBm, Pout=20dBm, PIM3=-10dBm
  - Gain = 20-(-6) = 26 dB
  - ΔdB = 20-(-10) = 30 dB
  - IIP3 = -6 + 30/2 = -6 + 15 = 9 dBm
  - OIP3 = 20 + 30/2 = 35 dBm

【MATCHED FILTER】
- For pulse s(t), matched filter impulse response: h(t) = s(T-t)
- If s(t) = A[U(t) - U(t-T/3)], then h(t) = A[U(t-2T/3) - U(t-T)]

【COHERENT OOK/2ASK OPTIMAL THRESHOLD】
- With unequal priors P0, P1: V_T = A/2 + (N0/A)*ln(P0/P1)

【DELTA MODULATION (DM)】
- DM SNR (dB) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0))
- Coherent BFSK: BER = 0.5 * erfc(√(Eb/(2*N0)))
- Non-coherent BFSK: BER = 0.5 * exp(-Eb/(2*N0))
- DPSK: BER = 0.5 * exp(-Eb/N0)

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
4. For numerical answers: Convert to BASE UNITS (Hz, W, s, bit/s)
5. For formula answers: Write the formula clearly

IMPORTANT: 
- For numerical answers: End with a clear number in base units
- For formula answers: Write the exact formula expression
"""