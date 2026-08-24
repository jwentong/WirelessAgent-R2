SC_ENSEMBLE_PROMPT = """
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

ENHANCED_QUESTION_TYPE_ANALYSIS_PROMPT = """Analyze the question and identify the EXACT type of answer expected.

Question Types:
1. ENTITY_NAME: Asking for the name of a person, place, or thing
   Example: "Who invented X?", "What is the name of Y?"
   
2. TYPE_DESCRIPTION: Asking for what TYPE or CATEGORY something belongs to
   Example: "What type of deity?", "What kind of animal?"
   
3. ATTRIBUTE_VALUE: Asking for a specific attribute or characteristic
   Example: "What distinction?", "What color?", "How many?"
   
4. LOCATION: Asking WHERE something is located
   Example: "Where is X located?", "In which city?"
   
5. ALTERNATIVE_NAME: Asking for another name or title
   Example: "Also known as?", "Also published as?"
   
6. COMPARISON_RESULT: Asking which one in a comparison
   Example: "Which has more?", "Which is bigger?"
   
7. DIFFERENCE_POINT: Asking HOW things differ (not just what they are)
   Example: "How do they differ?", "What's the difference?"
   
8. ACRONYM_EXPANSION: Asking what an acronym stands for
   Example: "What does ABC stand for?"
   
9. TEMPORAL_INFO: Asking about time, season, or period
   Example: "Which season?", "When did X happen?"

Instructions:
1. Identify which type above matches this question
2. If asking "what type/kind", answer must be a TYPE, not a specific instance
3. If asking "how differ", answer must be the DIFFERENCE POINT, not descriptions
4. If asking "also [verb] as", answer must be the ALTERNATIVE form
5. Output format: "[TYPE]: [One-sentence explanation of what to extract]"

Example:
Question: "What type of deity is Yemoja?"
Output: "TYPE_DESCRIPTION: Extract the category/type of deity (e.g., 'water deity'), not the deity's name."
"""

ANSWER_VALIDATION_PROMPT = """Question: {question}
Expected Answer Type: {expected_type}
Provided Answer: {answer}

Validation Rules:
1. If expected type is TYPE_DESCRIPTION, answer must be a category/type, not a specific name
2. If expected type is ALTERNATIVE_NAME, answer must be different from the name in question
3. If expected type is ACRONYM_EXPANSION, answer must be full words, not abbreviation
4. If expected type is DIFFERENCE_POINT, answer must be the difference, not full descriptions
5. If expected type is ENTITY_NAME, answer should be a proper name
6. If expected type is TEMPORAL_INFO and question asks "which season", convert "Season N" to ordinal word

Task: 
- If answer matches expected type: return it unchanged
- If answer doesn't match: extract the correct part or suggest correction
- Output format: "VALID: [answer]" or "CORRECTED: [corrected_answer]"

Provide only the validation result."""