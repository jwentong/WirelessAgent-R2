from pathlib import Path
from typing import List, Optional, Dict, Any

from scripts.formatter import BaseFormatter, XmlFormatter, TextFormatter
from workspace.WCHW.workflows.template.operator_an import *
from workspace.WCHW.workflows.template.op_prompt import *
from scripts.async_llm import AsyncLLM
from scripts.logs import logger


from scripts.operators import Operator, ReActAgent
from scripts.base_operator import BaseOperator
from scripts.tools import ToolRegistry

# RAG retriever for few-shot learning
try:
    from scripts.rag import get_retriever, reset_retriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logger.warning("RAG module not available")

# Note: Custom, ScEnsemble, ToolAgent, RAGRetriever are the 4 operators for WCHW


class Custom(BaseOperator):
    """Custom operator with flexible instruction-based prompting"""
    
    def __init__(self, llm: AsyncLLM, name: str = "Custom"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Input text"},
                "instruction": {"type": "string", "description": "Instruction for processing"}
            },
            "required": ["input", "instruction"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Generated response"}
            }
        }
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Helper method for LLM calls with formatting"""
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            response = await self.llm(prompt)
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None
    
    async def _execute(self, input: str, instruction: str, **kwargs) -> Dict:
        prompt = instruction + input
        response = await self._fill_node(GenerateOp, prompt, mode="single_fill")
        return response
    
    async def __call__(self, input, instruction):
        """Legacy compatibility"""
        result = await self._execute(input=input, instruction=instruction)
        return result


class ScEnsemble(BaseOperator):
    """
    Self-Consistency Ensemble operator
    
    Paper: Self-Consistency Improves Chain of Thought Reasoning in Language Models
    Link: https://arxiv.org/abs/2203.11171
    Paper: Universal Self-Consistency for Large Language Model Generation
    Link: https://arxiv.org/abs/2311.17311
    """

    def __init__(self, llm: AsyncLLM, name: str = "ScEnsemble"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "solutions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of candidate solutions"
                },
                "problem": {"type": "string", "description": "Original problem statement"}
            },
            "required": ["solutions", "problem"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string", "description": "Selected best solution"}
            }
        }
    
    async def _fill_node(self, op_class, prompt, mode=None, **extra_kwargs):
        """Helper method for LLM calls with formatting (inherited from Operator pattern)"""
        formatter = self._create_formatter(op_class, mode, **extra_kwargs)
        if formatter:
            response = await self.llm.call_with_format(prompt, formatter)
        else:
            response = await self.llm(prompt)
        if isinstance(response, dict):
            return response
        else:
            return {"response": response}
    
    def _create_formatter(self, op_class, mode=None, **extra_kwargs) -> Optional[BaseFormatter]:
        """Create appropriate formatter based on operation class and mode"""
        if mode == "xml_fill":
            return XmlFormatter.from_model(op_class)
        elif mode == "code_fill":
            function_name = extra_kwargs.get("function_name")
            return CodeFormatter(function_name=function_name)
        elif mode == "single_fill":
            return TextFormatter()
        else:
            return None

    async def _execute(self, solutions: List[str], problem: str, **kwargs) -> Dict:
        answer_mapping = {}
        solution_text = ""
        for index, solution in enumerate(solutions):
            answer_mapping[chr(65 + index)] = index
            solution_text += f"{chr(65 + index)}: \n{str(solution)}\n\n\n"

        prompt = SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=solution_text)
        response = await self._fill_node(ScEnsembleOp, prompt, mode="xml_fill")

        answer = response.get("solution_letter", "")
        answer = answer.strip().upper()

        return {"response": solutions[answer_mapping[answer]]}
    
    # Legacy compatibility
    async def __call__(self, solutions: List[str], problem: str):
        return await self._execute(solutions=solutions, problem=problem)


class ToolAgent(BaseOperator):
    """
    Tool-augmented Agent for WCHW (Wireless Communication Homework) problems
    
    Uses ReActAgent with domain-specific wireless communication tools plus
    general mathematical tools to solve wireless communication problems.
    
    Inherits from BaseOperator for automatic metrics collection and
    metadata introspection to support MCTS optimization.
    
    Tools available (6 core tools, optimized for WCHW):
    
    **WIRELESS-SPECIFIC (Domain-optimized)**:
    1. wireless_formula - Wireless communication formula library:
       - SNR/quantization formulas (SNR_dB = 1.76 + 6.02*n)
       - Carson's rule for FM bandwidth (BW = 2(f_m + Δf))
       - Shannon capacity (C = B*log2(1 + SNR))
       - AM power calculations (P_total = P_c(1 + m²/2))
       - FM SNR improvement (G_FM = 3β²(1+β))
       - PSK BER (0.5*erfc(sqrt(Eb/N0)))
       - Nyquist rate, PCM bitrate, raised-cosine bandwidth, matched filter SNR
    
    2. unit_converter - Unit conversions for wireless:
       - Frequency: Hz/kHz/MHz/GHz
       - Power: W/mW/dBm/dBW, dB/linear conversions
       - Data rate: bps/kbps/Mbps/Gbps
       - Voltage: V/mV/μV
    
    **GENERAL COMPUTATION**:
    3. python_code_solver - Generate and execute Python code for complex calculations
    4. calculator - Evaluate mathematical expressions using SymPy
    5. symbolic_solver - Solve equations symbolically for exact solutions
    
    **VALIDATION**:
    6. answer_type_validator - Validate and correct answer format (remove unwanted units/symbols)
    
    Features:
    - ReActAgent for tool-based reasoning (max 3 iterations)
    - Domain-specific wireless formulas for common calculations
    - Automatic unit conversion support
    - Intelligent fallback handling
    - WCHW-optimized strategy prompt
    - Removed geometry/combinatorics tools (not relevant for wireless communication)
    
    v4.0 Changes:
    - Removed WirelessFormulaLibrary (now in RAGRetriever context)
    - Removed UnitConverter, SymbolicSolver, AnswerValidator (simplified)
    - Only Programmer (python_code_solver) and RAG Retriever remain
    - Calculator tools removed (python_code_solver can handle calculations)
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "ToolAgent", react_strategy: str = None):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        
        # Initialize tool registry - v5.1 with Programmer + RAG Retriever
        self.tool_registry = ToolRegistry()
        
        # Register Programmer as the primary tool (Python code execution)
        from scripts.tools import ProgrammerTool
        programmer_tool = ProgrammerTool(llm)
        self.tool_registry.register(programmer_tool)
        
        # v5.1: Register telecom formula retriever (RAG tool)
        # Removed: calculator and telecom_calculator (python_code_solver handles calculations)
        try:
            from scripts.enhanced_tools import TelecomFormulaRetriever
            rag_tool = TelecomFormulaRetriever()
            self.tool_registry.register(rag_tool)
            logger.info("ToolAgent v5.1: Registered telecom_formula_retriever (RAG)")
        except ImportError as e:
            logger.warning(f"TelecomFormulaRetriever not available: {e}")
        
        logger.info(f"ToolAgent v5.1: Registered {len(self.tool_registry.tools)} tools (python_code_solver, telecom_formula_retriever)")
        
        # Use external strategy if provided, otherwise use default from op_prompt.py
        if react_strategy is None:
            react_strategy = REACT_STRATEGY_PROMPT
            logger.info("ToolAgent v5.0: Using default REACT_STRATEGY_PROMPT")
        else:
            logger.info(f"ToolAgent v5.0: Using custom react_strategy ({len(react_strategy)} chars)")
        
        # Initialize ReActAgent with strategy prompt
        self.react_agent = ReActAgent(
            llm, 
            self.tool_registry, 
            name="WCHW_ReAct",
            strategy_prompt=react_strategy  # Now optimizable via parameter!
        )
        
        logger.info(f"ToolAgent v5.0 initialized with {len(self.tool_registry.tools)} tools")
    
    def _get_input_schema(self) -> Dict:
        """Define input schema for ToolAgent"""
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The mathematical problem to solve"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of reasoning steps",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["problem"],
            "description": "ToolAgent accepts a math problem and optionally max_steps for reasoning depth"
        }
    
    def _get_output_schema(self) -> Dict:
        """Define output schema"""
        return {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer to the math problem"
                },
                "steps": {
                    "type": "array",
                    "description": "Tool calling history (thought-action-observation)",
                    "items": {"type": "object"}
                },
                "total_cost": {
                    "type": "number",
                    "description": "Total LLM cost incurred"
                },
                "used_fallback": {
                    "type": "boolean",
                    "description": "Whether fallback (non-tool) method was used"
                }
            },
            "description": "Returns answer with reasoning steps and metadata"
        }
    
    async def _execute(self, problem: str, max_steps: int = 1, **kwargs) -> Dict:
        """
        Use tools to solve the math problem with intelligent fallback
        
        Args:
            problem: Mathematical problem to solve
            max_steps: Maximum number of reasoning steps (default: 1)
        
        Returns:
            {
                "answer": str,
                "steps": List[Dict],  # Tool calling history
                "total_cost": float,
                "used_fallback": bool
            }
        """
        try:
            # Use ReActAgent for tool-based problem solving
            result = await self.react_agent(
                problem=problem,
                max_iterations=max_steps,
                verbose=False
            )
            
            answer = result.get("answer", "")
            steps = result.get("steps", [])
            
            return {
                "answer": answer,
                "steps": steps,
                "total_cost": self._extract_cost(result),
                "used_fallback": False
            }
            
        except Exception as e:
            logger.error(f"ToolAgent error: {e}, falling back to direct LLM solving")
            
            # Fallback: Use direct LLM solving with SOLVE_PROMPT
            try:
                prompt = PYTHON_CODE_VERIFIER_PROMPT.format(
                    problem=problem,
                    analysis="None",
                    feedback=""
                )
                response = await self.llm(prompt)
                
                return {
                    "answer": response if isinstance(response, str) else str(response),
                    "steps": [],
                    "total_cost": self._extract_cost({}),
                    "used_fallback": True
                }
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {
                    "answer": "Error in solving",
                    "steps": [],
                    "total_cost": 0.0,
                    "used_fallback": True
                }
    
    def _extract_cost(self, output) -> float:
        """Extract LLM cost"""
        if self.llm and hasattr(self.llm, 'get_usage_summary'):
            summary = self.llm.get_usage_summary()
            if 'total_cost' in summary:
                return summary['total_cost']
        return 0.0


# ============================================================================
# FORMULA LIBRARY (v5.0) - Categorized for problem-type matching
# ============================================================================

# Problem type → Formula mapping (key insight: RAG should identify type, then give specific formula)
FORMULA_BY_TYPE = {
    "bandwidth_raised_cosine": {
        "keywords": ["滚降", "升余弦", "raised cosine", "roll-off", "PSK.*带宽", "QAM.*带宽", "MPSK", "MQAM"],
        "formula": "B = Rs × (1 + α) / 2，其中 Rs = Rb / log2(M)",
        "warning": "注意：要除以2！不是 Rs×(1+α)"
    },
    "bandwidth_nrz": {
        "keywords": ["NRZ", "不归零码", "基带传输"],
        "formula": "B = Rb (第一零点带宽等于比特率)",
        "warning": None
    },
    "ber_bpsk_qpsk": {
        "keywords": ["BPSK", "QPSK", "误码率", "BER", "相干PSK"],
        "formula": "BER = 0.5 × erfc(√(Eb/N0))",
        "warning": None
    },
    "ber_bfsk_coherent": {
        "keywords": ["BFSK", "相干FSK", "coherent FSK"],
        "formula": "BER = 0.5 × erfc(√(Eb/N0 / 2))",
        "warning": "注意：Eb/N0 要除以2！"
    },
    "ber_bfsk_noncoherent": {
        "keywords": ["非相干", "non-coherent", "包络检测"],
        "formula": "BER = 0.5 × exp(-Eb/N0 / 2)",
        "warning": None
    },
    "ber_dpsk": {
        "keywords": ["DPSK", "差分"],
        "formula": "BER = 0.5 × exp(-Eb/N0)",
        "warning": None
    },
    "delta_modulation": {
        "keywords": ["增量调制", "DM", "delta modulation", "斜率过载"],
        "formula": "SNR_dB = -13.60 + 30×log10(fs/fm)",
        "warning": "这是DM公式，不要用PCM的 6.02n+1.76！"
    },
    "pcm_sqnr": {
        "keywords": ["PCM", "量化", "SQNR", "信噪比", "量化噪声", "quantization", "levels", "量化等级", "L =", "bits per sample"],
        "formula": "SQNR_dB = 6.02n + 1.76，其中 n = log2(L) 是量化位数，L是量化等级数",
        "warning": "仅用于PCM，不用于DM！如果给出量化等级L，先算 n = log2(L)"
    },
    "fm_bandwidth": {
        "keywords": ["FM", "调频", "Carson", "带宽"],
        "formula": "BW = 2(Δf + fm)，其中 β = Δf / fm",
        "warning": "如果幅度翻倍，Δf也翻倍"
    },
    "shannon_capacity": {
        "keywords": ["香农", "Shannon", "信道容量", "容量", "spectral efficiency", "频谱效率", "bit/s/Hz", "Nyquist signaling"],
        "formula": "C = B × log2(1 + SNR_linear)，其中 SNR_linear = 10^(SNR_dB/10)；频谱效率 η = C/B = log2(1+SNR)",
        "warning": None
    },
    "rayleigh_fading": {
        "keywords": ["瑞利", "Rayleigh", "电平通过率", "平均衰落时间"],
        "formula": "N_R = √(2π) × fD × ρ × exp(-ρ²)；τ = (exp(ρ²) - 1) / (ρ × fD × √(2π))",
        "warning": None
    },
    "error_correction": {
        "keywords": ["纠错", "检错", "汉明", "Hamming", "最小距离", "block code", "块码", "d_min", "detectable", "correctable", "分组码"],
        "formula": "检错能力 t_d = d_min - 1；纠错能力 t_c = floor((d_min - 1) / 2)",
        "warning": "注意区分：检错能力 vs 纠错能力！"
    },
}


class RAGRetriever(BaseOperator):
    """
    RAG-based Problem Solver for WCHW (v5.0)
    
    Core improvement: Problem Type → Formula Matching
    Instead of giving LLM all formulas, identify problem type and provide ONLY the relevant formula.
    
    Features:
    - Problem type detection via keyword matching
    - Targeted formula selection (not full library)
    - Optional few-shot examples from similar problems
    """
    
    def __init__(self, llm: AsyncLLM, name: str = "RAGRetriever"):
        super().__init__(llm, enable_metrics=True)
        self.name = name
        self.retriever = None
        self._init_retriever()
    
    def _init_retriever(self):
        """Initialize RAG retriever"""
        if not RAG_AVAILABLE:
            logger.warning("RAGRetriever: RAG module not available")
            return
        try:
            # Path: operator.py → template/ → workflows/ → WCHW/ → workspace/ → project_root/
            base_dir = Path(__file__).parent.parent.parent.parent.parent
            kb_path = base_dir / "data" / "datasets" / "wchw_validate.jsonl"
            if kb_path.exists():
                self.retriever = get_retriever(knowledge_base_path=str(kb_path), enhanced=True)
                logger.info(f"RAGRetriever: Loaded knowledge base from {kb_path}")
            else:
                logger.warning(f"RAGRetriever: Knowledge base not found at {kb_path}")
        except Exception as e:
            logger.error(f"RAGRetriever: Failed to initialize: {e}")
    
    def _detect_problem_type(self, problem: str) -> List[Dict]:
        """Detect problem type and return matching formulas (with scoring)"""
        import re
        problem_lower = problem.lower()
        scored_matches = []
        
        for type_name, info in FORMULA_BY_TYPE.items():
            match_count = 0
            for keyword in info["keywords"]:
                # Support regex patterns (e.g., "PSK.*带宽")
                if '.*' in keyword or '\\' in keyword:
                    if re.search(keyword.lower(), problem_lower):
                        match_count += 2  # Regex match scores higher
                elif keyword.lower() in problem_lower:
                    match_count += 1
            
            if match_count > 0:
                scored_matches.append({
                    "type": type_name,
                    "formula": info["formula"],
                    "warning": info.get("warning"),
                    "score": match_count
                })
        
        # Sort by score (descending) and return top matches
        scored_matches.sort(key=lambda x: x["score"], reverse=True)
        return scored_matches[:2]  # Return top 2 matches
    
    def _get_input_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "num_examples": {"type": "integer", "default": 2},
                "mode": {"type": "string", "enum": ["answer", "examples"], "default": "answer"}
            },
            "required": ["problem"]
        }
    
    def _get_output_schema(self) -> Dict:
        return {
            "type": "object",
            "properties": {
                "response": {"type": "string"},
                "formulas": {"type": "string"},
                "problem_types": {"type": "array"},
                "examples_used": {"type": "integer"}
            }
        }
    
    async def _execute(self, problem: str, num_examples: int = 2, mode: str = "answer", **kwargs) -> Dict:
        """
        v5.0: Problem Type Detection → Targeted Formula → Solution
        """
        # Step 1: Detect problem type and get matching formulas
        matched_formulas = self._detect_problem_type(problem)
        
        # Build formula context (only relevant formulas, not the entire library)
        formula_parts = []
        for mf in matched_formulas[:2]:  # Max 2 formulas to keep focused
            formula_parts.append(f"【公式】{mf['formula']}")
            if mf.get('warning'):
                formula_parts.append(f"【注意】{mf['warning']}")
        formula_context = "\n".join(formula_parts) if formula_parts else ""
        
        # Step 2: Get similar examples (optional, for format reference)
        examples_used = 0
        example_answer = ""
        if self.retriever:
            try:
                result = self.retriever.retrieve(problem)
                similar = result.get('similar_problems', [])[:num_examples]
                if similar:
                    examples_used = len(similar)
                    # Only extract answer format, not full solution
                    example_answer = similar[0].get('answer', similar[0].get('ground_truth', ''))
            except Exception as e:
                logger.warning(f"RAGRetriever: {e}")
        
        # Mode: examples - return formula guidance for other operators
        if mode == "examples":
            return {
                "response": "",
                "formulas": formula_context,
                "problem_types": [mf["type"] for mf in matched_formulas],
                "examples_used": examples_used,
                "example_answer": example_answer
            }
        
        # Mode: answer - solve with targeted formula
        prompt = RAG_TARGETED_SOLVE_PROMPT.format(
            formula_guidance=formula_context if formula_context else "根据问题类型选择合适的公式",
            problem=problem,
            answer_hint=f"参考答案格式：{example_answer}" if example_answer else ""
        )
        
        response = await self.llm(prompt)
        return {
            "response": response if isinstance(response, str) else str(response),
            "formulas": formula_context,
            "problem_types": [mf["type"] for mf in matched_formulas],
            "examples_used": examples_used
        }
    
    async def __call__(self, problem: str, num_examples: int = 2, mode: str = "answer", **kwargs):
        return await self._execute(problem=problem, num_examples=num_examples, mode=mode, **kwargs)




# RAG Prompt v5.1 - Targeted formula + Strict Output Format
RAG_TARGETED_SOLVE_PROMPT = """你是无线通信专家。请根据给定的公式解决问题。

{formula_guidance}

问题：{problem}

{answer_hint}

⚠️ 输出要求（非常重要！）：
1. 只输出最终数值答案，不要解释过程
2. 必须使用基本单位：
   - 频率用 Hz（不是 kHz/MHz），例如 36 kHz → 输出 36000
   - 功率用 W（不是 mW/μW），例如 0.5 mW → 输出 0.0005
   - 时间用 s（不是 ms/μs），例如 125 μs → 输出 0.000125
   - 速率用 bit/s（不是 kbit/s）
3. 不要输出单位符号，只输出纯数字
4. 不要使用 markdown 格式

答案（只输出数字）："""


# Legacy prompt for fallback
RAG_FALLBACK_PROMPT = """你是无线通信专家。

问题：{problem}

请解答此题，只输出最终答案（数值和单位）。

答案："""
