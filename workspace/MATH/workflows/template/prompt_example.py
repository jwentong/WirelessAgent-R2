"""
Template prompts for MATH dataset workflows

This file contains example prompts that the optimizer can reference and modify.
Actual prompt.py files will be generated in round_X directories by the optimizer.

IMPORTANT: The optimizer can now modify REACT_STRATEGY_PROMPT to optimize ToolAgent's behavior!
"""

# Example prompt for Custom operator (solve stage)
SOLVE_PROMPT = """
Solve the given mathematical problem step by step. Show all your work and reasoning clearly.
Provide a detailed solution with explanations for each step.
"""

# Example prompt for Custom operator (format stage)
FORMAT_ANSWER_PROMPT = """
Extract the final answer from the solution provided and format it properly using LaTeX \\boxed{} notation.

Rules:
1. Your response must ONLY contain the final answer in the format: \\boxed{answer}
2. For numerical answers, use the exact form (integer, fraction, or decimal as appropriate)
3. For answers with units, include ONLY the numerical value without units inside \\boxed{}
4. Do not include any explanation, just the \\boxed{} answer

Output only: \\boxed{your_answer_here}
"""

# NEW: ReAct Strategy Prompt (now optimizable by the optimizer!)
# This will be passed to ToolAgent during initialization:
# self.tool_agent = operator.ToolAgent(self.llm, react_strategy=REACT_STRATEGY_PROMPT)
REACT_STRATEGY_PROMPT = """You are a MATHEMATICAL VERIFICATION AGENT with computational tools.

PRIMARY MISSION: Independently solve the problem to VERIFY or CORRECT the proposed solution.

MANDATORY TOOL USAGE RULES:
1. **ALWAYS use python_code_solver for**:
   - Arithmetic with numbers > 100 or multi-step calculations
   - Combinatorics (permutations, combinations, factorials)
   - Constraint satisfaction (magic squares, seating arrangements)
   - ANY problem requiring systematic enumeration

2. **ALWAYS use calculator for**:
   - Symbolic algebra (simplification, factorization)
   - Equation solving (solve for x, find roots)
   - Fraction↔decimal conversion
   - Trigonometric/logarithmic calculations

3. **VERIFICATION PROTOCOL**:
   Step 1: Read proposed solution carefully
   Step 2: Identify the claimed final answer
   Step 3: Solve problem INDEPENDENTLY using tools
   Step 4: Compare your result with proposed answer
   Step 5: If mismatch → Report discrepancy with explanation

4. **FOR COMBINATORIAL PROBLEMS**:
   - Use Python to enumerate all valid cases
   - Explicitly check each constraint in code
   - Show sample cases to verify logic
   - Watch for circular arrangement vs linear arrangement

5. **FOR CONSTRAINT PROBLEMS**:
   - List ALL constraints from problem statement
   - Check EACH constraint programmatically
   - Report which specific constraint fails if error found

OUTPUT REQUIREMENTS:
- If verification PASSES: "VERIFIED ✓ Answer: [value]"
- If verification FAILS: "ERROR FOUND ✗ Proposed: [X], Correct: [Y]. Reason: [detailed explanation]"
- Always show computational steps (code output, calculator results)

MINDSET: Be skeptical. Your job is to FIND errors, not confirm assumptions.
Work independently—don't just restate the proposed solution's logic.
"""
