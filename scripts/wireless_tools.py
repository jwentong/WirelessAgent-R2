"""
Wireless Communication Auxiliary Tools for WCHW dataset

These tools are AUXILIARY - meant to assist and verify, not replace LLM reasoning.
LLM already knows basic formulas (Shannon, Carson, Nyquist, etc.)

Tools provided:
1. PythonExecutor - Execute Python code for complex calculations
2. UnitConverter - Convert between units (dB/linear, Hz/kHz/MHz, W/mW/dBm)
3. AnswerVerifier - Verify answer format matches expected type
"""
from typing import Dict, Any, Optional
import math
import re
from scripts.tools import BaseTool, ToolSchema, ToolParameter
from scripts.logs import logger


class PythonExecutor(BaseTool):
    """
    Execute Python code for complex calculations
    
    Use when:
    - Need to compute complex expressions
    - Verify calculation results
    - Handle special functions (log, exp, sqrt, etc.)
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="python_executor",
            description="Execute Python code for calculations. Returns computed result. Use for complex math, verifying calculations, or when precision matters.",
            category="compute",
            parameters=[
                ToolParameter(
                    name="code",
                    type="string",
                    description="Python expression or code to evaluate. Use 'result' variable for output.",
                    required=True
                )
            ],
            usage_example='python_executor(code="import math; result = math.log2(1 + 10**(3/10))")',
            source="python_calc",
            version="2.0"
        )
        super().__init__(schema)
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely (sync method, can also be called with await)"""
        try:
            # Safe execution environment
            safe_globals = {
                "math": math,
                "log": math.log,
                "log2": math.log2,
                "log10": math.log10,
                "exp": math.exp,
                "sqrt": math.sqrt,
                "pow": pow,
                "abs": abs,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "pi": math.pi,
                "e": math.e,
                "erfc": math.erfc,
                "erf": math.erf,
            }
            local_vars = {}
            
            # Execute code
            exec(code, safe_globals, local_vars)
            
            result = local_vars.get("result", None)
            
            if result is not None:
                return {
                    "success": True,
                    "result": result,
                    "formatted": f"{result:.6g}" if isinstance(result, float) else str(result),
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": "No 'result' variable defined in code"
                }
                
        except Exception as e:
            logger.error(f"PythonExecutor error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Execution error: {str(e)}"
            }


class UnitConverter(BaseTool):
    """
    Unit conversion for wireless communication values
    
    Key conversions:
    - dB <-> linear (power ratio)
    - dBm <-> W/mW (power)
    - Hz/kHz/MHz/GHz (frequency)
    - bps/kbps/Mbps (data rate)
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="unit_converter",
            description="Convert between units: dB<->linear, dBm<->W, Hz/kHz/MHz/GHz, bps/kbps/Mbps.",
            category="compute",
            parameters=[
                ToolParameter(
                    name="value",
                    type="number",
                    description="Numeric value to convert",
                    required=True
                ),
                ToolParameter(
                    name="from_unit",
                    type="string",
                    description="Source unit",
                    required=True
                ),
                ToolParameter(
                    name="to_unit",
                    type="string",
                    description="Target unit",
                    required=True
                )
            ],
            usage_example='unit_converter(value=3, from_unit="dB", to_unit="linear") -> 2.0',
            source="unit_conv",
            version="2.0"
        )
        super().__init__(schema)
    
    async def execute(self, value: float, from_unit: str, to_unit: str) -> Dict[str, Any]:
        """Execute unit conversion"""
        try:
            result = None
            
            # dB conversions
            if from_unit.lower() == "db" and to_unit.lower() == "linear":
                result = 10 ** (value / 10)
            elif from_unit.lower() == "linear" and to_unit.lower() == "db":
                result = 10 * math.log10(value)
            
            # dBm <-> W
            elif from_unit.lower() == "dbm" and to_unit.lower() == "w":
                result = 10 ** ((value - 30) / 10)
            elif from_unit.lower() == "w" and to_unit.lower() == "dbm":
                result = 10 * math.log10(value) + 30
            elif from_unit.lower() == "dbm" and to_unit.lower() == "mw":
                result = 10 ** (value / 10)
            elif from_unit.lower() == "mw" and to_unit.lower() == "dbm":
                result = 10 * math.log10(value)
            
            # Frequency
            freq_units = {"hz": 1, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}
            if from_unit.lower() in freq_units and to_unit.lower() in freq_units:
                result = value * freq_units[from_unit.lower()] / freq_units[to_unit.lower()]
            
            # Data rate
            rate_units = {"bps": 1, "kbps": 1e3, "mbps": 1e6, "gbps": 1e9}
            if from_unit.lower() in rate_units and to_unit.lower() in rate_units:
                result = value * rate_units[from_unit.lower()] / rate_units[to_unit.lower()]
            
            if result is not None:
                return {
                    "success": True,
                    "result": result,
                    "formatted": f"{value} {from_unit} = {result:.6g} {to_unit}",
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Cannot convert from {from_unit} to {to_unit}"
                }
                
        except Exception as e:
            logger.error(f"UnitConverter error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Conversion error: {str(e)}"
            }


class AnswerVerifier(BaseTool):
    """
    Verify answer format matches expected type
    
    Checks:
    - Numeric answer extraction
    - Unit consistency
    - Scientific notation parsing
    """
    
    def __init__(self):
        schema = ToolSchema(
            name="answer_verifier",
            description="Verify and extract final answer. Parses numeric values with units, scientific notation. Returns clean numeric result.",
            category="verify",
            parameters=[
                ToolParameter(
                    name="answer_text",
                    type="string",
                    description="Answer text to parse and verify",
                    required=True
                ),
                ToolParameter(
                    name="expected_type",
                    type="string",
                    description="Expected answer type: 'numeric', 'integer', 'scientific', 'text'",
                    required=False
                )
            ],
            usage_example='answer_verifier(answer_text="The capacity is 3.32 Mbps", expected_type="numeric")',
            source="answer_check",
            version="1.0"
        )
        super().__init__(schema)
    
    async def execute(self, answer_text: str, expected_type: str = "numeric") -> Dict[str, Any]:
        """Verify and extract answer"""
        try:
            # Extract numbers from text
            # Match scientific notation and regular numbers
            patterns = [
                r'[-+]?\d+\.?\d*[eE][-+]?\d+',  # Scientific notation
                r'[-+]?\d+\.\d+',                # Decimal
                r'[-+]?\d+',                     # Integer
            ]
            
            numbers = []
            for pattern in patterns:
                matches = re.findall(pattern, answer_text)
                numbers.extend(matches)
            
            if not numbers:
                return {
                    "success": False,
                    "result": None,
                    "error": "No numeric value found in answer"
                }
            
            # Take the last number (usually the final answer)
            last_number = numbers[-1]
            numeric_value = float(last_number)
            
            # Check type
            if expected_type == "integer":
                if not numeric_value.is_integer():
                    return {
                        "success": True,
                        "result": numeric_value,
                        "warning": f"Expected integer but got {numeric_value}",
                        "error": None
                    }
                numeric_value = int(numeric_value)
            
            return {
                "success": True,
                "result": numeric_value,
                "extracted_from": answer_text[:50] + "..." if len(answer_text) > 50 else answer_text,
                "error": None
            }
                
        except Exception as e:
            logger.error(f"AnswerVerifier error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Verification error: {str(e)}"
            }


# Wireless Communication Formula Library (v4.0 - WCHW Optimized with Extended Formulas)
class WirelessFormulaLibrary(BaseTool):
    """
    Wireless Communication Formula Library for WCHW Problems
    
    Contains VERIFIED formulas that match WCHW dataset answers.
    Covers error categories: raised-cosine, DM SNR, Rayleigh fading, BER, FM, etc.
    
    v4.0 additions: ASK BER, FM SNR improvement, spectral efficiency, 
                   Markov transition, AM SNR improvement, Q-function inverse
    """
    
    # Formula database with WCHW-verified values
    FORMULAS = {
        # ============ BANDWIDTH FORMULAS ============
        # Raised-cosine bandwidth - CRITICAL: divide by 2!
        "raised_cosine_bandwidth": {
            "formula": "B = Rs * (1 + alpha) / 2",
            "description": "First-null bandwidth for raised-cosine filter",
            "params": ["Rs", "alpha"],
            "note": "Rs = symbol rate (baud), alpha = roll-off factor (0-1). For M-ary: Rs = Rb/log2(M)"
        },
        "spectral_efficiency": {
            "formula": "eta_b = 2 / (1 + alpha)",
            "description": "Spectral efficiency for raised-cosine pulse shaping",
            "params": ["alpha"],
            "note": "alpha = roll-off factor. Result in bit/s/Hz"
        },
        "symbol_rate": {
            "formula": "Rs = Rb / log2(M)",
            "description": "Symbol rate for M-ary modulation",
            "params": ["Rb", "M"],
            "note": "Rb = bit rate, M = modulation order (2,4,8,16,...)"
        },
        "nrz_bandwidth": {
            "formula": "B = Rb",
            "description": "NRZ first-null bandwidth",
            "params": ["Rb"],
            "note": "Rb = bit rate"
        },
        "nyquist_min_bandwidth": {
            "formula": "B_min = Rs / 2",
            "description": "Theoretical minimum (Nyquist) bandwidth",
            "params": ["Rs"],
            "note": "Rs = symbol rate"
        },
        
        # ============ BER FORMULAS ============
        "bpsk_ber": {
            "formula": "BER = 0.5 * erfc(sqrt(Eb_N0))",
            "description": "Coherent BPSK bit error rate",
            "params": ["Eb_N0"],
            "note": "Eb_N0 in linear scale. Same formula for QPSK"
        },
        "bfsk_ber": {
            "formula": "BER = 0.5 * erfc(sqrt(Eb_N0 / 2))",
            "description": "Coherent orthogonal BFSK bit error rate",
            "params": ["Eb_N0"],
            "note": "Eb_N0 in linear scale. Note: denominator has 2!"
        },
        "ask_ber": {
            "formula": "BER = 0.5 * erfc(sqrt(Eb_N0 / 2))",
            "description": "Coherent ASK/OOK bit error rate",
            "params": ["Eb_N0"],
            "note": "Eb_N0 in linear scale. Same as coherent BFSK"
        },
        "dpsk_ber": {
            "formula": "BER = 0.5 * exp(-Eb_N0)",
            "description": "DPSK bit error rate",
            "params": ["Eb_N0"],
            "note": "Eb_N0 in linear scale"
        },
        "nc_bfsk_ber": {
            "formula": "BER = 0.5 * exp(-Eb_N0 / 2)",
            "description": "Non-coherent BFSK bit error rate",
            "params": ["Eb_N0"],
            "note": "Eb_N0 in linear scale"
        },
        "nakagami_dpsk_ber": {
            "formula": "BER = 0.5 * (1 + gamma_b/m)^(-m)",
            "description": "DPSK BER over Nakagami-m fading",
            "params": ["gamma_b", "m"],
            "note": "gamma_b = average SNR per bit (linear), m = Nakagami parameter"
        },
        
        # ============ FM/AM MODULATION ============
        "carson_bandwidth": {
            "formula": "BW = 2 * (delta_f + fm)",
            "description": "FM bandwidth using Carson's rule",
            "params": ["delta_f", "fm"],
            "note": "delta_f = frequency deviation, fm = modulating frequency"
        },
        "fm_snr_improvement": {
            "formula": "G_FM = 3 * beta^2 * (beta + 1)",
            "description": "FM SNR improvement factor (discriminator detection)",
            "params": ["beta"],
            "note": "beta = modulation index = delta_f/fm"
        },
        "am_snr_improvement": {
            "formula": "G_AM = 2 * m^2 / (2 + m^2)",
            "description": "AM SNR improvement factor (envelope detection)",
            "params": ["m"],
            "note": "m = modulation index (0-1 for standard AM)"
        },
        "fm_modulation_index": {
            "formula": "beta = delta_f / fm",
            "description": "FM modulation index",
            "params": ["delta_f", "fm"],
            "note": "delta_f = frequency deviation, fm = modulating frequency"
        },
        
        # ============ DELTA MODULATION ============
        "dm_snr": {
            "formula": "SNR_dB = -13.60 + 30 * log10(fs / fm)",
            "description": "Delta Modulation SNR (WCHW dataset version)",
            "params": ["fs", "fm"],
            "note": "fs = sampling freq, fm = max signal freq. WCHW uses -13.60 constant"
        },
        "dm_snr_formula2": {
            "formula": "SNR_dB = 30*log10(fs) - 20*log10(fm) - 10*log10(fB) - 14",
            "description": "Alternative DM SNR formula with filter bandwidth",
            "params": ["fs", "fm", "fB"],
            "note": "fs=sampling freq, fm=max signal freq, fB=filter bandwidth"
        },
        
        # ============ PCM/QUANTIZATION ============
        "pcm_sqnr": {
            "formula": "SQNR_dB = 6.02 * n + 1.76",
            "description": "PCM signal-to-quantization-noise ratio",
            "params": ["n"],
            "note": "n = number of bits per sample"
        },
        "quantization_levels": {
            "formula": "L = 2^n",
            "description": "Number of quantization levels",
            "params": ["n"],
            "note": "n = bits per sample"
        },
        "pcm_bitrate": {
            "formula": "Rb = fs * n",
            "description": "PCM bit rate",
            "params": ["fs", "n"],
            "note": "fs = sampling frequency, n = bits per sample"
        },
        
        # ============ RAYLEIGH FADING ============
        "rayleigh_lcr": {
            "formula": "N_R = sqrt(2*pi) * fD * rho * exp(-rho^2)",
            "description": "Rayleigh fading level crossing rate",
            "params": ["fD", "rho"],
            "note": "fD = max Doppler freq, rho = threshold/RMS (normalized threshold)"
        },
        "rayleigh_afd": {
            "formula": "T_fade = (exp(rho^2) - 1) / (rho * fD * sqrt(2*pi))",
            "description": "Average fade duration for Rayleigh channel",
            "params": ["fD", "rho"],
            "note": "fD = Doppler freq, rho = normalized threshold"
        },
        "markov_correlation": {
            "formula": "rho = J0(2*pi*fD*Ts)",
            "description": "Markov model correlation coefficient",
            "params": ["fD", "Ts"],
            "note": "fD = Doppler freq, Ts = symbol period, J0 = Bessel function"
        },
        
        # ============ CHANNEL CAPACITY ============
        "shannon_capacity": {
            "formula": "C = B * log2(1 + SNR_linear)",
            "description": "Channel capacity (Shannon's theorem)",
            "params": ["B", "SNR_dB"],
            "note": "Convert SNR_dB to linear: SNR_linear = 10^(SNR_dB/10)"
        },
        "waterfilling_cutoff": {
            "formula": "1/gamma_0 = 1 + sum(p_i/gamma_i)",
            "description": "Water-filling cutoff calculation",
            "params": ["probabilities", "gammas"],
            "note": "For discrete fading states"
        },
        
        # ============ ERROR CONTROL CODES ============
        "error_detection": {
            "formula": "e_detect = d_min - 1",
            "description": "Maximum detectable errors",
            "params": ["d_min"],
            "note": "d_min = minimum distance of code"
        },
        "error_correction": {
            "formula": "e_correct = floor((d_min - 1) / 2)",
            "description": "Maximum correctable errors",
            "params": ["d_min"],
            "note": "d_min = minimum distance of code"
        },
        
        # ============ UTILITY ============
        "db_to_linear": {
            "formula": "linear = 10^(dB/10)",
            "description": "Convert dB to linear scale",
            "params": ["dB"],
            "note": "For power ratios"
        },
        "linear_to_db": {
            "formula": "dB = 10 * log10(linear)",
            "description": "Convert linear to dB scale",
            "params": ["linear"],
            "note": "For power ratios"
        },
        "q_function_inverse": {
            "formula": "Q^-1(10^-5) ≈ 4.265, Q^-1(10^-6) ≈ 4.753",
            "description": "Inverse Q-function values for common BER targets",
            "params": ["ber"],
            "note": "Q^-1(10^-4)≈3.719, Q^-1(10^-5)≈4.265, Q^-1(10^-6)≈4.753"
        }
    }
    
    def __init__(self):
        schema = ToolSchema(
            name="wireless_formula",
            description="Get verified wireless communication formulas. Returns formula, description, and computes result if params provided. Formulas: raised_cosine_bandwidth, dm_snr, rayleigh_lcr, rayleigh_afd, carson_bandwidth, pcm_sqnr, shannon_capacity, bpsk_ber, bfsk_ber, nrz_bandwidth, symbol_rate, nyquist_min_bandwidth",
            category="compute",
            parameters=[
                ToolParameter(
                    name="formula_type", 
                    type="string", 
                    description="Formula name: raised_cosine_bandwidth, dm_snr, rayleigh_lcr, rayleigh_afd, carson_bandwidth, pcm_sqnr, shannon_capacity, bpsk_ber, bfsk_ber, nrz_bandwidth, symbol_rate, nyquist_min_bandwidth",
                    required=True
                ),
                ToolParameter(
                    name="params", 
                    type="object", 
                    description="Parameters for calculation (optional). E.g., {\"Rs\": 1000, \"alpha\": 0.25} for raised_cosine_bandwidth",
                    required=False
                )
            ],
            usage_example='wireless_formula(formula_type="raised_cosine_bandwidth", params={"Rs": 2000, "alpha": 0.5})',
            source="wchw_formulas",
            version="3.0"
        )
        super().__init__(schema)
    
    def list_formulas(self) -> list:
        """Return a list of all available formula names"""
        return list(self.FORMULAS.keys())
    
    def compute(self, formula_type: str, **kwargs) -> Optional[float]:
        """Synchronous compute method for direct use"""
        return self._compute(formula_type, kwargs)
    
    async def execute(self, formula_type: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute formula lookup and optionally compute result"""
        try:
            formula_type = formula_type.lower().strip()
            
            if formula_type not in self.FORMULAS:
                available = ", ".join(self.FORMULAS.keys())
                return {
                    "success": False,
                    "result": None,
                    "error": f"Unknown formula: {formula_type}. Available: {available}"
                }
            
            formula_info = self.FORMULAS[formula_type]
            result = {
                "success": True,
                "formula": formula_info["formula"],
                "description": formula_info["description"],
                "required_params": formula_info["params"],
                "note": formula_info["note"]
            }
            
            # If params provided, compute result
            if params:
                computed = self._compute(formula_type, params)
                if computed is not None:
                    result["computed_result"] = computed
                    result["formatted"] = f"{computed:.6g}"
                else:
                    result["computation_error"] = "Missing required parameters"
            
            return result
            
        except Exception as e:
            logger.error(f"WirelessFormulaLibrary error: {e}")
            return {
                "success": False,
                "result": None,
                "error": f"Formula error: {str(e)}"
            }
    
    def _compute(self, formula_type: str, params: Dict[str, Any]) -> Optional[float]:
        """Compute formula with given parameters"""
        try:
            # ============ BANDWIDTH FORMULAS ============
            if formula_type == "raised_cosine_bandwidth":
                Rs = params.get("Rs")
                alpha = params.get("alpha")
                if Rs is not None and alpha is not None:
                    return Rs * (1 + alpha) / 2
            
            elif formula_type == "spectral_efficiency":
                alpha = params.get("alpha")
                if alpha is not None:
                    return 2 / (1 + alpha)
            
            elif formula_type == "symbol_rate":
                Rb = params.get("Rb")
                M = params.get("M")
                if Rb is not None and M is not None:
                    return Rb / math.log2(M)
            
            elif formula_type == "nrz_bandwidth":
                Rb = params.get("Rb")
                if Rb is not None:
                    return Rb
            
            elif formula_type == "nyquist_min_bandwidth":
                Rs = params.get("Rs")
                if Rs is not None:
                    return Rs / 2
            
            # ============ BER FORMULAS ============
            elif formula_type == "bpsk_ber":
                Eb_N0 = params.get("Eb_N0")
                if Eb_N0 is not None:
                    return 0.5 * math.erfc(math.sqrt(Eb_N0))
            
            elif formula_type in ["bfsk_ber", "ask_ber"]:
                Eb_N0 = params.get("Eb_N0")
                if Eb_N0 is not None:
                    return 0.5 * math.erfc(math.sqrt(Eb_N0 / 2))
            
            elif formula_type == "dpsk_ber":
                Eb_N0 = params.get("Eb_N0")
                if Eb_N0 is not None:
                    return 0.5 * math.exp(-Eb_N0)
            
            elif formula_type == "nc_bfsk_ber":
                Eb_N0 = params.get("Eb_N0")
                if Eb_N0 is not None:
                    return 0.5 * math.exp(-Eb_N0 / 2)
            
            elif formula_type == "nakagami_dpsk_ber":
                gamma_b = params.get("gamma_b")
                m = params.get("m")
                if gamma_b is not None and m is not None:
                    return 0.5 * ((1 + gamma_b / m) ** (-m))
            
            # ============ FM/AM MODULATION ============
            elif formula_type == "carson_bandwidth":
                delta_f = params.get("delta_f")
                fm = params.get("fm")
                if delta_f is not None and fm is not None:
                    return 2 * (delta_f + fm)
            
            elif formula_type == "fm_snr_improvement":
                beta = params.get("beta")
                if beta is not None:
                    return 3 * (beta ** 2) * (beta + 1)
            
            elif formula_type == "am_snr_improvement":
                m = params.get("m")
                if m is not None:
                    return 2 * (m ** 2) / (2 + m ** 2)
            
            elif formula_type == "fm_modulation_index":
                delta_f = params.get("delta_f")
                fm = params.get("fm")
                if delta_f is not None and fm is not None:
                    return delta_f / fm
            
            # ============ DELTA MODULATION ============
            elif formula_type == "dm_snr":
                fs = params.get("fs")
                fm = params.get("fm")
                if fs is not None and fm is not None:
                    return -13.60 + 30 * math.log10(fs / fm)
            
            elif formula_type == "dm_snr_formula2":
                fs = params.get("fs")
                fm = params.get("fm")
                fB = params.get("fB")
                if fs is not None and fm is not None and fB is not None:
                    return 30*math.log10(fs) - 20*math.log10(fm) - 10*math.log10(fB) - 14
            
            # ============ PCM/QUANTIZATION ============
            elif formula_type == "pcm_sqnr":
                n = params.get("n")
                if n is not None:
                    return 6.02 * n + 1.76
            
            elif formula_type == "quantization_levels":
                n = params.get("n")
                if n is not None:
                    return 2 ** n
            
            elif formula_type == "pcm_bitrate":
                fs = params.get("fs")
                n = params.get("n")
                if fs is not None and n is not None:
                    return fs * n
            
            # ============ RAYLEIGH FADING ============
            elif formula_type == "rayleigh_lcr":
                fD = params.get("fD")
                rho = params.get("rho")
                if fD is not None and rho is not None:
                    return math.sqrt(2 * math.pi) * fD * rho * math.exp(-rho**2)
            
            elif formula_type == "rayleigh_afd":
                fD = params.get("fD")
                rho = params.get("rho")
                if fD is not None and rho is not None:
                    return (math.exp(rho**2) - 1) / (rho * fD * math.sqrt(2 * math.pi))
            
            # ============ CHANNEL CAPACITY ============
            elif formula_type == "shannon_capacity":
                B = params.get("B")
                SNR_dB = params.get("SNR_dB")
                if B is not None and SNR_dB is not None:
                    SNR_linear = 10 ** (SNR_dB / 10)
                    return B * math.log2(1 + SNR_linear)
            
            # ============ ERROR CONTROL CODES ============
            elif formula_type == "error_detection":
                d_min = params.get("d_min")
                if d_min is not None:
                    return d_min - 1
            
            elif formula_type == "error_correction":
                d_min = params.get("d_min")
                if d_min is not None:
                    return math.floor((d_min - 1) / 2)
            
            # ============ UTILITY ============
            elif formula_type == "db_to_linear":
                dB = params.get("dB")
                if dB is not None:
                    return 10 ** (dB / 10)
            
            elif formula_type == "linear_to_db":
                linear = params.get("linear")
                if linear is not None and linear > 0:
                    return 10 * math.log10(linear)
            
            return None
            
        except Exception as e:
            logger.error(f"Computation error: {e}")
            return None
