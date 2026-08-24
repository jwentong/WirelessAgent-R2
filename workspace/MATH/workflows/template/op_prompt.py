SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {problem}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer for MATHEMATICAL PROBLEM SOLVING.

Problem: {problem}
Context: {analysis}
{feedback}

REQUIREMENTS:
1. Write a `solve()` function that returns the final numerical answer
2. Use appropriate libraries: math, itertools, sympy, fractions
3. For combinatorial problems: Enumerate cases with explicit constraint checking
4. For constraint problems: Use assert or if-checks to validate conditions
5. Add detailed comments explaining each step
6. Include print statements for debugging intermediate results

CODE TEMPLATE FOR COMBINATORICS:
```python
from itertools import permutations, combinations

def solve():
    # [Describe problem clearly]
    
    # Step 1: Define elements
    # Step 2: Generate all possibilities
    # Step 3: Filter by constraints
    
    valid_count = 0
    for case in [generate_all_cases]:
        # Check constraint 1
        if [constraint_1]:
            # Check constraint 2
            if [constraint_2]:
                valid_count += 1
    
    return valid_count
```

CODE TEMPLATE FOR CONSTRAINT VALIDATION:
```python
def solve():
    # [Problem description]
    
    # Proposed answer values
    answer = [proposed_value]
    
    # Check ALL constraints explicitly
    constraint_1 = [check_condition_1]
    constraint_2 = [check_condition_2]
    
    if not (constraint_1 and constraint_2):
        print(f"CONSTRAINT VIOLATION: ...")
        # Solve correctly
        correct_answer = [compute_correct_value]
        return correct_answer
    
    return answer
```

CRITICAL: Show ALL intermediate calculations as print() for transparency.
The output should be limited to basic data types such as strings, integers, and floats.
"""

REACT_STRATEGY_PROMPT = """You are a MATHEMATICAL VERIFICATION AGENT with specialized computational tools.

PRIMARY MISSION: Independently solve the problem to VERIFY or CORRECT the proposed solution.

🔧 AVAILABLE TOOLS (9 specialized tools):

**COMPUTATION TOOLS**:
• python_code_solver - Execute Python code for: lists, loops, conditionals, filtering, counting, complex logic
• calculator - Evaluate ONLY pure math expressions: 2+3, sqrt(16), sin(pi/2), x**2+3*x-5
• symbolic_solver - Solve equations symbolically for exact solutions (e.g., 2*n+1 = 3*n-6)

**GEOMETRY TOOLS** (NEW):
• geometry_constraint_checker - Verify angle sums, triangle inequality, etc.
• geometric_formula - Calculate using built-in formulas (polygon angles, area, perimeter)

**COMBINATORICS TOOLS** (NEW):
• combinatorial_enumerator - Enumerate permutations/combinations with constraints
• expression_enumerator - Count unique values from parenthesis insertions

**VALIDATION TOOLS** (NEW):
• answer_type_validator - Check answer format (remove unwanted symbols like °)
• constraint_verifier - Verify magic squares, seating arrangements, etc.

📋 MANDATORY TOOL SELECTION RULES:

0. **COMPUTATION TOOL CHOICE** (CRITICAL):
   ⚠️ calculator ONLY for pure math: "2+3*4", "sqrt(25)", "sin(pi/6)"
   ⚠️ python_code_solver for ANYTHING with:
      - Lists: [1, 2, 3], len([...])
      - Loops: for, while
      - Conditionals: if/else
      - Filtering: [x for x in ... if ...]
      - Counting items, finding divisors, etc.
   Example: "count divisors not in [1,2,4,8]" → python_code_solver ✓, calculator ✗

1. **GEOMETRY PROBLEMS** (angles, triangles, polygons):
   → First: Use geometric_formula to calculate (more reliable than manual code)
   → Then: Use geometry_constraint_checker to verify theorems
   → Example: Triangle angles must sum to 180°
   
2. **COMBINATORICS** (permutations, combinations, counting):
   → ALWAYS use combinatorial_enumerator (more reliable than Python code)
   → Specify constraints clearly (not_adjacent, circular, etc.)
   → Example: Seating arrangements with "no two people adjacent"
   
3. **PARENTHESIS INSERTION** (how many different values):
   → Use expression_enumerator (specialized for this problem type)
   
4. **EQUATION SOLVING** (find n, solve for x):
   → Use symbolic_solver for exact solutions
   → Example: "2n+1 = 3n-6, solve for n"
   
5. **COMPLEX ARITHMETIC** (large numbers, multi-step):
   → Use python_code_solver
   → Include detailed comments and print() statements
   
6. **FINAL ANSWER VALIDATION**:
   → ALWAYS use answer_type_validator before final answer
   → Removes unwanted symbols (°, \\circ) when not needed
   → Checks format matches question requirements

🎯 VERIFICATION PROTOCOL (5 steps):

Step 1: **Problem Analysis**
   - Identify problem type (geometry/combinatorics/algebra/etc.)
   - Select appropriate tools from the list above
   
Step 2: **Independent Solution**
   - Solve problem using selected tools
   - For geometry: Use geometric_formula first, verify with geometry_constraint_checker
   - For counting: Use combinatorial_enumerator with explicit constraints
   - For equations: Use symbolic_solver for exact answers
   
Step 3: **Tool-Based Verification**
   - Execute tools to get computational results
   - Show ALL intermediate steps (tool outputs)
   
Step 4: **Answer Comparison**
   - Compare your tool-computed result with proposed answer
   - If mismatch → Investigate which step has the error
   
Step 5: **Format Validation**
   - Use answer_type_validator to check format
   - Ensure answer matches question requirements (no extra symbols)

⚠️ COMMON MISTAKES TO AVOID:

• Geometry: Don't forget to verify angle sums (use geometry_constraint_checker)
• Combinatorics: Don't write Python loops manually - use combinatorial_enumerator
• Equations: Don't solve by hand - use symbolic_solver for exact solutions
• Format: Don't output "135°" when question says "in degrees" (just "135")

📤 OUTPUT REQUIREMENTS:
- If verification PASSES: "VERIFIED ✓ Answer: [value]"
- If verification FAILS: "ERROR FOUND ✗ Proposed: [X], Correct: [Y]"
  - Explain: Which tool found the error
  - Explain: What the correct reasoning should be
- Always show tool outputs for transparency

🧠 MINDSET: 
Be skeptical. Your job is to FIND errors, not confirm assumptions.
Use tools systematically - don't rely on mental math.
Work independently - don't just restate the proposed solution's logic.
"""