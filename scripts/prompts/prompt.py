# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : prompts of operators

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

FORMAT_PROMPT = """
For the question described as {problem_description},
please extract a short and concise answer contains only one word/few words from the following solution: {solution}.
Make sure there are no additional comments or explanations in your response.
"""

SC_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

PYTHON_CODE_VERIFIER_PROMPT = """
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model.
"""


REFLECTION_ON_PUBLIC_TEST_PROMPT = """
Given a code problem and a python code solution which failed to pass test or execute, you need to analyze the reason for the failure and propose a better code solution.: 
### problem
{problem}

### Code Solution
{solution}

### Execution Result
{exec_pass}

#### Failed Test Case
{test_fail}

Please provide a reflection on the failed test cases and code solution, followed by a better code solution without any additional text or test cases.
"""

MD_ENSEMBLE_PROMPT = """
Given the question described as follows: {question}
Several solutions have been generated to address the given question. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the solution that is more capable of solving the problem compared to other solutions, as this is crucial for problem-solving.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

REVIEW_PROMPT = """
Given a problem and a thoughtful solution, your task is to using critical thinking (questioning) to review the solution's correctness and provide a review result in boolean format.

problem: {problem}
solution: {solution}

If you are more than 95 percent confident that the final answer is incorrect, please return False and give a feedback for the error. Otherwise, please return True and give a explanation for the correctness.
"""

REVISE_PROMPT = """
Given a problem and a thoughtful solution which is just reviewed as incorrect, your task is to revise the solution to solve the question and ensure the final code solution is wrapped with ```python```.

problem: {problem}
solution: {solution}
feedback: {feedback}

Ensure the output code is self-contained, and without any additional text or test cases.
"""

REACT_AGENT_FIXED_PROTOCOL = """
IMPORTANT: Your response MUST be in valid XML format with ALL required tags. Do NOT omit any tags.

OUTPUT FORMAT (MANDATORY):

If you want to use a tool, output EXACTLY this format:
<thought>Your reasoning about what to do next</thought>
<action_type>use_tool</action_type>
<tool_name>name of the tool</tool_name>
<tool_args>{"arg1": "value1", "arg2": "value2"}</tool_args>

If you want to provide final answer, output EXACTLY this format:
<thought>Your reasoning about why you can answer now</thought>
<action_type>final_answer</action_type>
<final_answer>Your CONCISE answer to the question (MUST NOT BE EMPTY)</final_answer>

STRICT REQUIREMENTS:
- ALWAYS include <thought> and <action_type> tags
- If action_type is "use_tool", MUST include <tool_name> and <tool_args>
- If action_type is "final_answer", MUST include <final_answer> WITH ACTUAL CONTENT
- CRITICAL: <final_answer> tag MUST contain the actual answer value, NOT be empty!
- tool_args MUST be a valid JSON object (dictionary) with the following JSON rules:
  * Use DOUBLE QUOTES for keys and string values: {"key": "value"}
  * NO single quotes allowed: {'key': 'value'} is INVALID
  * NO trailing commas: {"a": 1, "b": 2,} is INVALID
  * NO Python expressions in values: {"value": 1/(2*1)} is INVALID - compute first: {"value": 0.5}
  * All numeric values must be CALCULATED NUMBERS, not expressions
  * Empty dict is allowed: {}
  * Example VALID: {"expression": "2+3*4", "bandwidth_hz": 500000, "timeout": 5}
  * Example INVALID: {'expression': '2+3*4'} (single quotes)
  * Example INVALID: {"expression": "2+3*4",} (trailing comma)
  * Example INVALID: {{"expression": "2+3*4"}} (double braces)
  * Example INVALID: {"bandwidth_hz": 1/(2*1)} (Python expression - should be 0.5)
- Do NOT leave any tags empty or missing
"""
