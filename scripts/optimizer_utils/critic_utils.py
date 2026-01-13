import json
import re
import os
from scripts.logs import logger

class CriticUtils:
    """
    Smart Critic that considers workflow maturity before giving recommendations.
    
    Key principle: HIGH-SCORING workflows should receive CONSERVATIVE recommendations.
    LOW-SCORING workflows need STRUCTURAL changes.
    """
    
    # Score thresholds for recommendation aggressiveness
    HIGH_SCORE_THRESHOLD = 0.65  # Above this: only minor prompt tweaks
    MID_SCORE_THRESHOLD = 0.50   # Between this and HIGH: can suggest structure changes
    # Below MID: major changes allowed
    
    def __init__(self, root_path: str):
        self.root_path = root_path

    def _get_workflow_complexity(self, round_number: int) -> dict:
        """Analyze the current workflow structure to avoid over-engineering."""
        graph_path = os.path.join(self.root_path, "workflows", f"round_{round_number}", "graph.py")
        
        complexity = {
            "has_toolagent": False,
            "has_scensemble": False,
            "has_conditional": False,  # if/else branches
            "num_steps": 1,
            "num_custom_calls": 0,
        }
        
        if not os.path.exists(graph_path):
            return complexity
            
        try:
            with open(graph_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            complexity["has_toolagent"] = "tool_agent" in code.lower() or "toolagent" in code.lower()
            complexity["has_scensemble"] = "scensemble" in code.lower() or "sc_ensemble" in code.lower()
            complexity["has_conditional"] = bool(re.search(r'\bif\b.*:', code))
            complexity["num_custom_calls"] = len(re.findall(r'await\s+self\.custom\(', code))
            
            # Count approximate steps (await calls in __call__)
            call_section = re.search(r'async def __call__\(self.*?\):(.*?)(?=\n    async def|\Z)', code, re.DOTALL)
            if call_section:
                complexity["num_steps"] = len(re.findall(r'\bawait\b', call_section.group(1)))
                
        except Exception as e:
            logger.warning(f"Could not analyze workflow complexity: {e}")
            
        return complexity

    def _analyze_errors(self, round_number: int) -> dict:
        """Categorize errors from log file."""
        log_path = os.path.join(self.root_path, "workflows", f"round_{round_number}", "log.json")
        
        result = {
            "total": 0,
            "format_errors": 0,
            "unit_errors": 0,
            "value_errors": 0,
            "examples": []  # Store a few examples for context
        }
        
        if not os.path.exists(log_path):
            return result
            
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except Exception:
            return result
            
        if not logs:
            return result
            
        result["total"] = len(logs)
        unit_pattern = re.compile(r'\d+\s*[a-zA-Z]+')
        
        for i, log in enumerate(logs[:50]):  # Limit to first 50 for speed
            extracted = str(log.get('extracted_output', ''))
            model_out = str(log.get('model_output', ''))
            right_ans = str(log.get('right_answer', ''))
            
            error_type = "value"
            
            # Format error detection
            if unit_pattern.search(model_out) and (not extracted or extracted == "None"):
                error_type = "format"
            elif unit_pattern.search(extracted):
                error_type = "format"
            else:
                # Unit mismatch detection
                try:
                    nums_extracted = re.findall(r"[-+]?\d*\.\d+|\d+", extracted)
                    nums_right = re.findall(r"[-+]?\d*\.\d+|\d+", right_ans)
                    
                    if nums_extracted and nums_right:
                        val_extracted = float(nums_extracted[0])
                        val_right = float(nums_right[0])
                        
                        if val_right != 0:
                            ratio = val_extracted / val_right
                            if 0.99 < abs(ratio) < 1.01:
                                continue  # Actually correct
                            elif (0.9 < abs(ratio * 1000) < 1.1 or 0.9 < abs(ratio / 1000) < 1.1 or
                                  0.9 < abs(ratio * 1e6) < 1.1 or 0.9 < abs(ratio / 1e6) < 1.1):
                                error_type = "unit"
                except Exception:
                    if not extracted or extracted == "None":
                        error_type = "format"
                        
            if error_type == "format":
                result["format_errors"] += 1
            elif error_type == "unit":
                result["unit_errors"] += 1
            else:
                result["value_errors"] += 1
                
            # Store examples
            if len(result["examples"]) < 3:
                result["examples"].append({
                    "type": error_type,
                    "extracted": extracted[:100],
                    "expected": right_ans[:100]
                })
                
        return result

    def _get_exploration_status(self, round_number: int) -> dict:
        """
        Check how many times this round has been explored and with what results.
        This helps detect "saturated" nodes that shouldn't be modified further.
        """
        experience_path = os.path.join(self.root_path, "workflows", "processed_experience.json")
        
        status = {
            "n_success": 0,
            "n_failure": 0,
            "n_neutral": 0,
            "is_saturated": False,
            "failed_modifications": []
        }
        
        if not os.path.exists(experience_path):
            return status
            
        try:
            with open(experience_path, 'r', encoding='utf-8') as f:
                experience = json.load(f)
                
            round_exp = experience.get(str(round_number), {})
            status["n_success"] = len(round_exp.get("success", {}))
            status["n_failure"] = len(round_exp.get("failure", {}))
            status["n_neutral"] = len(round_exp.get("neutral", {}))
            
            # Extract failed modification keywords for guidance
            for child_round, data in round_exp.get("failure", {}).items():
                mod = data.get("modification", "")
                # Extract key action words
                if "toolagent" in mod.lower():
                    status["failed_modifications"].append("ToolAgent changes")
                if "conditional" in mod.lower() or "if/else" in mod.lower():
                    status["failed_modifications"].append("Conditional logic")
                if "3-step" in mod.lower() or "three-step" in mod.lower():
                    status["failed_modifications"].append("Multi-step workflow")
                if "enhance" in mod.lower():
                    status["failed_modifications"].append("Enhancement attempts")
                    
            status["failed_modifications"] = list(set(status["failed_modifications"]))
            
            # Node is saturated if: 2+ failures and no successes
            n_total = status["n_success"] + status["n_failure"] + status["n_neutral"]
            if n_total >= 2 and status["n_failure"] >= 2 and status["n_success"] == 0:
                status["is_saturated"] = True
                
        except Exception as e:
            logger.warning(f"Could not load exploration status: {e}")
            
        return status

    def analyze(self, round_number: int, current_score: float = 0.0) -> str:
        """
        Generate a SMART critic report that considers:
        1. Current score (high score = conservative changes)
        2. Workflow complexity (already complex = don't add more)
        3. Error patterns
        4. Exploration history (avoid repeating failed patterns)
        """
        complexity = self._get_workflow_complexity(round_number)
        errors = self._analyze_errors(round_number)
        exploration = self._get_exploration_status(round_number)
        
        # Determine recommendation aggressiveness
        if current_score >= self.HIGH_SCORE_THRESHOLD:
            aggressiveness = "CONSERVATIVE"
        elif current_score >= self.MID_SCORE_THRESHOLD:
            aggressiveness = "MODERATE"
        else:
            aggressiveness = "AGGRESSIVE"
            
        # Override to ULTRA-CONSERVATIVE if node is saturated
        if exploration["is_saturated"]:
            aggressiveness = "ULTRA-CONSERVATIVE"
            
        report = f"""
===============================================================
SMART CRITIC REPORT (Round {round_number})
Score: {current_score:.4f} | Recommendation Mode: {aggressiveness}
===============================================================

[ERROR ANALYSIS]
  Total Errors: {errors['total']}
  - Format/Extraction: {errors['format_errors']}
  - Unit Mismatch: {errors['unit_errors']}  
  - Calculation/Value: {errors['value_errors']}

[CURRENT WORKFLOW COMPLEXITY]
  - Steps: {complexity['num_steps']}
  - Uses ToolAgent: {'Yes' if complexity['has_toolagent'] else 'No'}
  - Uses ScEnsemble: {'Yes' if complexity['has_scensemble'] else 'No'}
  - Has Conditionals: {'Yes' if complexity['has_conditional'] else 'No'}
  - Custom Calls: {complexity['num_custom_calls']}

[EXPLORATION HISTORY]
  - Previous Attempts: {exploration['n_success'] + exploration['n_failure'] + exploration['n_neutral']}
  - Successful Modifications: {exploration['n_success']}
  - Failed Modifications: {exploration['n_failure']}
  - Node Saturated: {'[WARNING] YES' if exploration['is_saturated'] else 'No'}
"""
        if exploration["failed_modifications"]:
            report += f"  - Failed Patterns: {', '.join(exploration['failed_modifications'])}\n"

        report += "\n[RECOMMENDATIONS]\n"
        
        if aggressiveness == "ULTRA-CONSERVATIVE":
            # Saturated node: almost no modifications should be made
            report += f"""
[STOP] NODE SATURATED - MINIMAL CHANGES ONLY!

This round has been tried {exploration['n_failure']} times with all failures.
The optimizer should consider selecting a DIFFERENT parent round.

If you must modify this round, ONLY change:
- A single word in the prompt
- Punctuation or formatting
- Nothing else!

[FORBIDDEN] ABSOLUTELY FORBIDDEN (all previously failed):
"""
            for pattern in exploration["failed_modifications"]:
                report += f"- {pattern}\n"
                
        elif aggressiveness == "CONSERVATIVE":
            # High score: DON'T change structure, only minor tweaks
            report += """
[WARNING] HIGH SCORE DETECTED - BE VERY CAREFUL!

This workflow is already performing well. Only make MINIMAL changes:

[ALLOWED]
- Minor prompt wording improvements
- Add/remove one line of instruction
- Clarify unit conversion instructions

[FORBIDDEN]
- Adding new operators (ToolAgent, ScEnsemble)
- Removing existing operators
- Changing workflow structure (adding/removing steps)
- Adding if/else conditional logic
- Changing ToolAgent from 'verification' to 'calculation' role

The current structure works. Trust it.
"""
        elif aggressiveness == "MODERATE":
            # Medium score: can suggest single changes
            if complexity['has_conditional']:
                report += "\n[WARNING] Conditional logic detected and score is mediocre. Consider simplifying.\n"
            
            if errors['format_errors'] > errors['value_errors']:
                report += "\n- Focus on OUTPUT FORMAT in prompts (emphasize '只输出数字')\n"
            elif errors['unit_errors'] > 5:
                report += "\n- Improve unit conversion instructions\n"
                if not complexity['has_toolagent']:
                    report += "- Consider adding ToolAgent for VERIFICATION (not calculation)\n"
            else:
                report += "\n- Improve reasoning/formula instructions in prompt\n"
                
            report += """
[FORBIDDEN]
- Adding if/else conditional branches
- Making workflow more than 3 steps
- Adding multiple new operators at once
"""
        else:
            # Low score: can suggest structural changes
            report += """
Low score detected. Structural changes may be needed:

Suggestions based on error pattern:
"""
            if errors['format_errors'] > errors['value_errors']:
                report += "- Add strict output formatting in prompt\n"
            if errors['value_errors'] > errors['format_errors']:
                if not complexity['has_toolagent']:
                    report += "- Consider adding ToolAgent for calculation VERIFICATION\n"
                else:
                    report += "- ToolAgent already present - check if it's being used correctly\n"
                    
            report += """
[STILL FORBIDDEN]
- Adding if/else conditional branches (proven harmful)
- Making ToolAgent the PRIMARY calculator (it should VERIFY, not calculate)
- Adding RAGRetriever (proven harmful on this dataset)
"""
        
        # Add examples if available
        if errors['examples']:
            report += "\n[ERROR EXAMPLES]\n"
            for ex in errors['examples'][:2]:
                report += f"  [{ex['type'].upper()}] Got: {ex['extracted'][:50]}... Expected: {ex['expected'][:50]}...\n"
        
        return report
