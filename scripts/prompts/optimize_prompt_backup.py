WORKFLOW_OPTIMIZE_PROMPT = """You are building a Graph and corresponding Prompt to jointly solve {type} problems. 
Referring to the given graph and prompt, which forms a basic example of a {type} solution approach, 
please reconstruct and optimize them. You can add, modify, or delete nodes, parameters, or prompts. Include your 
single modification in XML tags in your reply. Ensure they are complete and correct to avoid runtime failures. When 
optimizing, you can incorporate critical thinking methods like review, revise, ensemble (generating multiple answers through different/similar prompts, then voting/integrating/checking the majority to obtain a final answer), selfAsk, etc. Consider 
Python's loops (for, while, list comprehensions), conditional statements (if-elif-else, ternary operators), 
or machine learning techniques (e.g., linear regression, decision trees, neural networks, clustering). The graph 
complexity should not exceed 10. Use logical and control flow (IF-ELSE, loops) for a more enhanced graphical 
representation.Ensure that all the prompts required by the current graph from prompt_custom are included.Exclude any other prompts.
Output the modified graph and all the necessary Prompts in prompt_custom (if needed).
The prompt you need to generate is only the one used in `prompt_custom.XXX` within Custom. Other methods already have built-in prompts and are prohibited from being generated. Only generate those needed for use in `prompt_custom`; please remove any unused prompts in prompt_custom.
the generated prompt must not contain any placeholders.
Considering information loss, complex graphs may yield better results, but insufficient information transmission can omit the solution. It's crucial to include necessary context during the process."""


WORKFLOW_INPUT = """
Here is a graph and the corresponding prompt (prompt only related to the custom method) that performed excellently in a previous iteration (maximum score is 1). You must make further optimizations and improvements based on this graph. The modified graph must differ from the provided example, and the specific differences should be noted within the <modification>xxx</modification> section.\n
<sample>
    <experience>{experience}</experience>
    <modification>(such as:add /delete /modify/ ...)</modification>
    <score>{score}</score>
    <graph>{graph}</graph>
    <prompt>{prompt}</prompt>(only prompt_custom)
    <operator_description>{operator_description}</operator_description>
</sample>
Below are the logs of some results with the aforementioned Graph that performed well but encountered errors, which can be used as references for optimization:
{log}

First, provide optimization ideas. **Only one detail point can be modified at a time**, and no more than 5 lines of code may be changed per modification—extensive modifications are strictly prohibited to maintain project focus!
When introducing new functionalities in the graph, please make sure to import the necessary libraries or modules yourself, except for operator, prompt_custom, create_llm_instance, and CostManage, which have already been automatically imported.
**Under no circumstances should Graph output None for any field.**
Use custom methods to restrict your output format, rather than using code (outside of the code, the system will extract answers based on certain rules and score them).
It is very important to format the Graph output answers, you can refer to the standard answer format in the log.
You do not need to manually import prompt_custom or operator to use them; they are already included in the execution environment.

**IMPORTANT - Using Custom Operators**:
The `operator` module contains your custom operators and is already imported in the template (e.g., `import workspace.HotpotQA.workflows.template.operator as operator`).
- Always use operators with the `operator.` prefix: `operator.ToolAgent(...)`, `operator.Custom(...)`, `operator.AnswerGenerate(...)`, etc.
- NEVER write `from operator import ...` as this will import Python's built-in operator module instead of your custom operators.
- All operators are accessed via the `operator` namespace that was imported for you.
"""

WORKFLOW_CUSTOM_USE = """\nHere's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.

**⚠️ EMPIRICALLY PROVEN BEST STRATEGIES for WCHW Dataset (Score: 0.34-0.40)**:

Based on extensive error analysis, here are the KEY improvements that work:

**🔑 ERROR ANALYSIS INSIGHTS**:
1. **Context-dependent questions** (0% accuracy): Problems referencing "previous problem", "same data", "above"
2. **Quantization/PCM** (0% accuracy): A-law, μ-law, PCM encoding
3. **Modulation** (4% accuracy): FM, AM, deviation, Carson's rule
4. **Bandwidth** (3.6% accuracy): Nyquist, raised-cosine, symbol rate
5. **Unit conversion errors** (15% of all errors): kHz↔Hz, dB↔linear

**✅ PROVEN WINNING STRATEGY (Round 13 - Score: 0.34-0.39)**:

**Strategy A: Context-Dependent Question Detection**
```python
CONTEXT_KEYWORDS = ['previous problem', 'same data', 'above', 'continuing from', 'same system']

def _is_context_dependent(self, problem: str) -> bool:
    return any(kw in problem.lower() for kw in CONTEXT_KEYWORDS)

# In __call__:
if self._is_context_dependent(problem):
    # Solve with explicit assumptions since context is missing
    assumption_prompt = f"This problem references previous context. Make reasonable assumptions and state them.\\n{problem}"
    solution = await self.custom(input=assumption_prompt, instruction=prompt_custom.SOLVE_PROMPT)
    return solution['response']
```

**Strategy B: Topic-Specific Knowledge Injection**
```python
# In prompt.py - define specialized knowledge bases
QUANTIZATION_KNOWLEDGE = \"\"\"
A-LAW 13-segment: S[ABC][WXYZ], segment k step size = 2^(k-1)
PCM bit rate: R_b = n × f_s, where f_s ≥ 2f_max
SQNR = 6.02n + 1.76 dB
\"\"\"

MODULATION_KNOWLEDGE = \"\"\"
FM Carson's Rule: BW = 2(Δf_max + f_m)
Modulation index β = Δf_max / f_m
After ×N multiplier: Δf_new = N × Δf_old
\"\"\"

BANDWIDTH_KNOWLEDGE = \"\"\"
Raised-cosine: BW = Rs(1 + α)/2 baseband, BW = Rs(1 + α) passband
Nyquist: R_s = 2B maximum
\"\"\"

# In graph.py - inject relevant knowledge
def _get_topic_knowledge(self, problem: str) -> str:
    p = problem.lower()
    if any(kw in p for kw in ['quantiz', 'pcm', 'a-law']):
        return prompt_custom.QUANTIZATION_KNOWLEDGE
    if any(kw in p for kw in ['fm', 'modulation', 'deviation']):
        return prompt_custom.MODULATION_KNOWLEDGE
    if any(kw in p for kw in ['bandwidth', 'nyquist', 'raised-cosine']):
        return prompt_custom.BANDWIDTH_KNOWLEDGE
    return ""
```

**Strategy C: Unit Conversion Guidance in Prompt**
```python
UNIT_GUIDE = \"\"\"
CRITICAL UNIT RULES:
- Match answer units to question (if asks kHz, answer in kHz)
- 1 GHz = 1000 MHz = 10^6 kHz = 10^9 Hz
- dB conversions: SNR_dB = 10·log₁₀(SNR_linear)
- Probability: always 0 to 1 (not percentage)
\"\"\"
```

**Strategy D: Selective Precision Verification**
```python
PRECISION_KEYWORDS = ['db', 'snr', 'log', 'quantiz', 'modulation', 'bandwidth', 'capacity']

if any(kw in problem.lower() for kw in PRECISION_KEYWORDS):
    # Verify calculation with enhanced prompt
    verify_prompt = f"Verify: {solution}\\nCheck formula, units, calculation."
    verification = await self.custom(input=verify_prompt, instruction=prompt_custom.VERIFY_PROMPT)
    return verification['response']
```

**📋 COMPLETE RECOMMENDED WORKFLOW**:
```python
async def __call__(self, problem: str):
    # A: Handle context-dependent questions
    if self._is_context_dependent(problem):
        # Use assumptions
        ...
    
    # B: Get topic-specific knowledge
    topic_knowledge = self._get_topic_knowledge(problem)
    
    # C: Get RAG few-shot examples
    few_shot = ""
    if self.retriever:
        rag_result = await self.retriever(problem=problem, num_examples=2, mode='examples')
        if rag_result.get('examples'):
            few_shot = f"Examples:\\n{rag_result['examples']}\\n\\n"
    
    # D: Build enhanced input
    enhanced_input = f"{few_shot}{topic_knowledge}\\nProblem: {problem}"
    
    # E: Solve
    solution = await self.custom(input=enhanced_input, instruction=prompt_custom.SOLVE_PROMPT)
    
    # F: Selective verification for precision-critical
    if self._needs_precision_check(problem):
        verification = await self.custom(input=f"Verify: {solution['response']}", instruction=prompt_custom.VERIFY_PROMPT)
        return verification['response']
    
    return solution['response']
```

**❌ AVOID THESE (Proven ineffective)**:
1. ToolAgent as primary solver (worse than direct LLM)
2. Multi-step chains (error accumulation)
3. Ensemble methods (noise > signal)
4. Complex conditionals without clear purpose

**Available Operators**:
- `Custom`: Primary solver with prompt engineering
- `RAGRetriever`: Few-shot examples (use mode='examples')
- `ToolAgent`: ONLY for verification, not solving

**Key Insight**: The evaluator extracts the LAST number. Ensure response ends with: FINAL_ANSWER: [number]
"""

WORKFLOW_TEMPLATE = """from typing import Literal
import workspace.{dataset}.workflows.template.operator as operator
import workspace.{dataset}.workflows.round_{round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

{graph}
"""
