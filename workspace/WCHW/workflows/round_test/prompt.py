# WCHW Round Test - Formula-enhanced RAG + Simplified ToolAgent
# Hypothesis: Formulas in RAG context > Formulas as tool calls

SOLVE_WITH_FORMULAS_PROMPT = """You are a telecommunications expert. Use the formula library provided to solve this problem.

=== SOLUTION APPROACH ===
1. **Identify** the problem type and what is asked
2. **Find matching formula** from the FORMULA LIBRARY provided
3. **Extract parameters** from the problem statement
4. **Calculate step by step** - show your work
5. **State final answer** with correct units

CRITICAL REMINDERS:
- BER for BFSK/ASK: denominator has factor of 2!
- Raised-cosine bandwidth: divide by 2! (B = Rs*(1+α)/2)
- DM uses different SNR formula than PCM!
- Always convert dB to linear before computing BER!

Provide the final answer as a clear number.
"""

ANSWER_EXTRACTION_PROMPT = """Extract the final numerical answer from the solution.

INSTRUCTIONS:
1. Find the final answer from the solution
2. Output ONLY the numerical value in simplest form
3. NO units, explanations, or extra text
4. Use decimal notation (0.001 not 1e-3)

Examples:
- "d_min = 4" → 4
- "BER = 10^(-6)" → 0.000001
- "bandwidth = 6750 Hz" → 6750

Output: Just the number.
"""

VERIFICATION_PROMPT = """You have Python available to verify calculations.

Problem: {problem}
Proposed Solution: {solution}

Use Python to:
1. Verify the formula selection is correct
2. Check the numerical calculation
3. If there's an error, compute the correct answer

Output only the final verified numerical answer.
"""
