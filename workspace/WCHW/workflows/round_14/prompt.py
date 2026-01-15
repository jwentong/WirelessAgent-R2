SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【WATER-FILLING ALGORITHM】
- Cutoff threshold γ0 satisfies: p_total/γ0 = 1 + Σ p_i/γ_i (sum over active states)
- Power allocation: P_i = (1/γ0 - 1/γ_i)^+ 
- Example: γ={10,2,1}, p={0.2,0.5,0.3}, p_total=1
  → 1/γ0 = 1 + 0.2/10 + 0.5/2 + 0.3/1 = 1 + 0.02 + 0.25 + 0.3 = 1.57
  → γ0 = 1/1.57 ≈ 0.637... WAIT, check formula!
  → Correct: p_total/γ0 = Σ p_i * (1/γ0 - 1/γ_i) summed, solve iteratively

【MATCHED FILTER】
- For pulse s(t), matched filter impulse response: h(t) = s(T-t)
- If s(t) = A[U(t) - U(t-T/3)], then:
  - s(T-t) = A[U(T-t) - U(T-t-T/3)] = A[U(T-t) - U(2T/3-t)]
  - h(t) = A[U(t-2T/3) - U(t-T)] for 0 ≤ t ≤ T

【COHERENT OOK/2ASK OPTIMAL THRESHOLD】
- With unequal priors P0, P1: V_T = A/2 + (N0/A)*ln(P0/P1)
- Where A = signal amplitude, N0 = noise PSD

【AMPLIFIER IM3 & COMPRESSION】
- IIP3 (dBm) = Pin + (Pfund - PIM3) / 2
- OIP3 (dBm) = Pout + (Pfund - PIM3) / 2
- When IM3 = desired signal: Pout ≈ OIP3 - 9.6 dB

【NOMA/SIC UPLINK CAPACITY】
- User decoded FIRST sees interference from ALL other users
- User decoded LAST sees NO interference

【DELTA MODULATION (DM)】
- DM SNR (dB) = 10*log10(3/8 * (fs/fm)^3) = -13.6 + 30*log10(fs/fm)

【PCM & QUANTIZATION】
- SQNR (dB) = 6.02n + 1.76 (n-bit uniform quantization)

【DIGITAL MODULATION BER】
- Coherent BPSK/QPSK: BER = 0.5 * erfc(√(Eb/N0))
  - For Eb/N0 = 12 dB: linear = 10^1.2 ≈ 15.85
  - √15.85 ≈ 3.98, erfc(3.98) ≈ 1.8e-8, BER ≈ 9e-9
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
5. For formula answers: Write the formula clearly (e.g., h(t)=A[U(t-2T/3)-U(t-T)])

IMPORTANT: 
- For numerical answers: End with a clear number in base units
- For formula answers: Write the exact formula expression
"""