SOLVE_PROMPT = """You are a telecommunications expert. Solve this problem step by step.

=== CRITICAL FORMULAS ===

【MAP DETECTION THRESHOLD】
- For 2PSK with prior probabilities P1, P2 and amplitude A:
- MAP threshold: γ = (N0/2A) * ln(P1/P2)
- Where N0 is noise power spectral density

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

【FM MODULATION】
- Carson's Rule: BW = 2(Δf + fm)
- Modulation index: β = Δf/fm

【BANDWIDTH】
- NRZ first-null: B = Rb
- Raised-cosine: B = Rs(1+α)/2, where Rs = Rb/log2(M)

【SHANNON CAPACITY】
- C = B * log2(1 + SNR_linear)
- SNR_linear = 10^(SNR_dB/10)

【BLOCK CODE】
- (n,k) code: n = total bits, k = data bits
- Parity bits = n - k
- Code rate R = k/n (this is the primary answer)

【CENTER FREQUENCY】
- Center frequency = (f_low + f_high) / 2
- MUST output in Hz (e.g., 1.9 GHz = 1900000000 Hz)

=== UNIT CONVERSION (CRITICAL!) ===
ALWAYS convert to base units before outputting:
- 1.9 GHz → 1900000000 (Hz)
- 200 MHz → 200000000 (Hz)
- 36 kHz → 36000 (Hz)
- 0.5 mW → 0.0005 (W)
- 10 ms → 0.01 (s)

=== OUTPUT RULES ===
1. If multiple values are asked (e.g., parity bits AND code rate), output the RATE/RATIO/PROBABILITY (decimal value)
2. Always use base units: Hz, W, s, bit/s
3. Output only the final number, no units

=== SOLUTION APPROACH ===
1. Identify given values and what is asked
2. Select the correct formula
3. Perform step-by-step calculations
4. Convert to base units
5. State the final numerical answer clearly

IMPORTANT: End your solution with a clear final answer as a number.
"""