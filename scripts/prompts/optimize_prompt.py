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

**🚨🚨🚨 CRITICAL: SCORE-BASED OPTIMIZATION RULES 🚨🚨🚨**

**YOUR CURRENT SCORE IS {score}**

📊 **OPTIMIZATION AGGRESSIVENESS BY SCORE**:

| Score Range | Strategy | What You CAN Do | What You CANNOT Do |
|-------------|----------|-----------------|-------------------|
| **≥0.65** (HIGH) | CONSERVATIVE | Minor prompt tweaks only | NO structural changes, NO adding operators |
| **0.50-0.65** (MEDIUM) | MODERATE | Single structural change | NO if/else logic, NO 4+ steps |
| **<0.50** (LOW) | AGGRESSIVE | Major restructuring | Still no RAGRetriever, no if/else |

⚠️ **IF YOUR SCORE IS ≥0.65**: 
The workflow is already working well! Making big changes will likely HURT performance.
Only make TINY prompt improvements. DO NOT:
- Add new operators (ToolAgent, ScEnsemble)
- Remove existing operators
- Change the workflow structure
- Add if/else conditional logic

**📊 OPTIMIZATION PRIORITY** (based on error analysis):
1. **OUTPUT FORMAT IS CRITICAL**: 35% of errors are due to answer extraction failures
   - Solution: Modify prompt to output ONLY a pure number (no units, no text)
   - Example: Output "36000" instead of "36 kHz" or "**36000 Hz**"
   
2. **UNIT CONVERSION**: 9% of errors are unit mismatches
   - Solution: Always convert to base units (Hz, W, s, bit/s) before outputting
   - Example: 36 kHz → 36000, 0.5 mW → 0.0005
   
3. **SIMPLIFY WORKFLOW**: Complex multi-step workflows cause information loss
   - Solution: Use 1-2 Custom calls maximum, not 4-5
   - Simpler is better!

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

WORKFLOW_CUSTOM_USE = '''
Here's an example of using the `custom` method in graph:
```
# You can write your own prompt in <prompt>prompt_custom</prompt> and then use it in the Custom method in the graph
response = await self.custom(input=problem, instruction=prompt_custom.XXX_PROMPT)
# You can also concatenate previously generated string results in the input to provide more comprehensive contextual information.
# response = await self.custom(input=problem+f"xxx:{xxx}, xxx:{xxx}", instruction=prompt_custom.XXX_PROMPT)
# The output from the Custom method can be placed anywhere you need it, as shown in the example below
solution = await self.generate(problem=f"question:{problem}, xxx:{response['response']}")
```
Note: In custom, the input and instruction are directly concatenated(instruction+input), and placeholders are not supported. Please ensure to add comments and handle the concatenation externally.

**🔥🔥🔥 CRITICAL: BALANCED OPTIMIZATION STRATEGY 🔥🔥🔥**

**AVAILABLE OPERATORS** (Use strategically):

1. **Custom** (Primary): Direct LLM reasoning. Best for straightforward problems.
   - Usage: `response = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)`
   
2. **ToolAgent** (For Complex Calculations): Python code execution for precise calculations.
   - Usage: `result = await self.tool_agent(problem=problem, max_steps=3)`
   - Returns: `result['answer']` (the computed answer)
   - Best for: unit conversions, complex formulas (erfc, log), numerical verification
   - **STRATEGY**: Use when LLM often gets calculations wrong
   
3. **ScEnsemble** (For Higher Accuracy): Voting among multiple solutions.
   - Usage: `final = await self.sc_ensemble(solutions=[sol1, sol2, sol3], problem=problem)`
   - Returns: `final['response']` (voted answer)
   - Best for: when you have 3+ different solutions to compare

**🔧 ToolAgent内置工具** (自动可用，无需额外代码):

ToolAgent v5.2 每次问题只调用一个工具：

| 工具名 | 用途 | 适用场景 |
|--------|------|----------|
| `telecom_formula_retriever` | 提高理解能力 | 不熟悉概念/公式，需要查询 |
| `python_code_solver` | 验证/精确计算 | 理解问题后，需要精确计算 |

**📌 工具选择规则**:
- 🤔 "不理解题目/公式" → `telecom_formula_retriever` (检索后用推理回答)
- ✅ "理解题目，需要计算" → `python_code_solver` (精确计算)
- ⚠️ 每次问题只用一个工具

**🎯 RECOMMENDED WORKFLOW PATTERNS**:

**Pattern A: Simple (Best for most problems)**
```python
solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
return solution['response'], cost
```

**Pattern B: Custom + Code Verification (For calculation-heavy problems)**
```python
# Step 1: LLM solves the problem
solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
# Step 2: ToolAgent verifies with Python code
verification = await self.tool_agent(problem=f"Verify this calculation: {problem}. Expected answer: {solution['response']}", max_steps=2)
return verification['answer'], cost
```

**Pattern C: Ensemble Voting (For maximum accuracy at higher cost)**
```python
solutions = []
for i in range(3):
    sol = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
    solutions.append(sol['response'])
final = await self.sc_ensemble(solutions=solutions, problem=problem)
return final['response'], cost
```

**⚠️ DO NOT**:
- Use more than 3 Custom calls (information loss)
- Add RAGRetriever (introduces noise)
- Make overly complex multi-step workflows

**🔧 RECOMMENDED SIMPLE WORKFLOW (Proven: 0.68 → 0.70+)**:
```python
class Workflow:
    def __init__(self, name, llm_config, dataset):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        # Single-step: solve and output pure number
        solution = await self.custom(input=problem, instruction=prompt_custom.SOLVE_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
```

**📝 CRITICAL PROMPT REQUIREMENTS** (include in your SOLVE_PROMPT):
```
你是无线通信专家。请解决以下问题。

⚠️ 输出要求（非常重要！）：
1. 只输出最终数值答案，不要解释过程
2. 必须使用基本单位：
   - 频率用 Hz（不是 kHz/MHz）
   - 功率用 W（不是 mW/μW）
   - 时间用 s（不是 ms/μs）
   - 速率用 bit/s（不是 kbit/s）
3. 不要输出单位符号，只输出数字
4. 示例：如果答案是 36 kHz，输出 36000

问题：{problem}

答案（只输出数字）：
```

**⚠️ COMMON MISTAKES TO AVOID**:
1. ❌ Complex workflow with 4+ Custom calls → Information loss
2. ❌ Output like "36 kHz" or "36000 Hz" → Should be just "36000"
3. ❌ Output with markdown (**bold**, $$latex$$) → Just plain number
4. ❌ Multiple verification steps → Adds cost without improving accuracy

**📊 FORMULA REFERENCE (for domain knowledge)**:

| Problem Type | Formula | Common Mistake |
|--------------|---------|----------------|
| Raised-cosine bandwidth | B = Rs×(1+α)/2, Rs=Rb/log2(M) | Forgetting to divide by 2 |
| PCM SQNR | SQNR_dB = 6.02n + 1.76 | Using for DM (wrong!) |
| Delta Modulation SNR | SNR_dB = -13.6 + 30×log10(fs/fm) | Using PCM formula |
| Coherent BFSK BER | 0.5×erfc(√(Eb/2N0)) | Forgetting factor of 2 |
| Shannon capacity | C = B×log2(1+SNR_linear) | Using dB instead of linear |

**Available Operators**:
- `Custom`: Primary solver for direct LLM reasoning
- `ToolAgent`: Python code execution for precise calculations (use for complex math)
- `ScEnsemble`: Voting for higher accuracy (3x cost)

**⛔ DO NOT USE**:
- `RAGRetriever`: Introduces noise, confuses the model

**🎯 KEY INSIGHT**: Use Custom for reasoning, ToolAgent for calculation verification, ScEnsemble for voting when needed.
'''

WORKFLOW_TEMPLATE = """from typing import Literal
import workspace.{dataset}.workflows.template.operator as operator
import workspace.{dataset}.workflows.round_{round}.prompt as prompt_custom
from scripts.async_llm import create_llm_instance


from scripts.evaluator import DatasetType

{graph}
"""
