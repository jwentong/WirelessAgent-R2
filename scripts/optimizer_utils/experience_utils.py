import json
import os
from collections import defaultdict

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file

class ExperienceUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def load_experience(self, path=None, mode: str = "Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        experience_data = defaultdict(lambda: {"score": None, "success": {}, "failure": {}, "neutral": {}})

        for round_dir in os.listdir(rounds_dir):
            if os.path.isdir(os.path.join(rounds_dir, round_dir)) and round_dir.startswith("round_"):
                round_path = os.path.join(rounds_dir, round_dir)
                try:
                    round_number = int(round_dir.split("_")[1])
                    json_file_path = os.path.join(round_path, "experience.json")
                    if os.path.exists(json_file_path):
                        data = read_json_file(json_file_path, encoding="utf-8")
                        father_node = data["father node"]

                        if experience_data[father_node]["score"] is None:
                            experience_data[father_node]["score"] = data["before"]

                        if data["succeed"] is True:
                            experience_data[father_node]["success"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        elif data["succeed"] is False:
                            experience_data[father_node]["failure"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                        else:
                            # Neutral - within noise range
                            experience_data[father_node]["neutral"][round_number] = {
                                "modification": data["modification"],
                                "score": data["after"],
                            }
                except Exception as e:
                    logger.info(f"Error processing {round_dir}: {str(e)}")

        experience_data = dict(experience_data)

        output_path = os.path.join(rounds_dir, "processed_experience.json")
        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(experience_data, outfile, indent=4, ensure_ascii=False)

        logger.info(f"Processed experience data saved to {output_path}")
        return experience_data

    def format_experience(self, processed_experience, sample_round):
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            experience = f"Original Score: {experience_data['score']}\n"
            experience += "These are conclusions from past experiments:\n\n"
            
            # Only prohibit actual failures (significant degradation)
            if experience_data.get("failure"):
                experience += "**FAILED approaches (DO NOT repeat):**\n"
                for key, value in experience_data["failure"].items():
                    experience += f"- {value['modification']} (Score dropped to: {value['score']})\n"
            
            # Note successful approaches to learn from
            if experience_data.get("success"):
                experience += "\n**SUCCESSFUL approaches (learn from these):**\n"
                for key, value in experience_data["success"].items():
                    experience += f"- {value['modification']} (Score improved to: {value['score']})\n"
            
            # Neutral approaches are okay to try variations of
            if experience_data.get("neutral"):
                experience += "\n**NEUTRAL approaches (minor variations okay):**\n"
                for key, value in experience_data["neutral"].items():
                    experience += f"- {value['modification']} (Score: {value['score']}, within noise)\n"
            
            experience += "\n\nNote: Avoid repeating failed approaches. Build on successful patterns. Neutral results suggest the direction may be worth exploring with different parameters."
        else:
            experience = f"No experience data found for round {sample_round}."
        return experience

    def check_modification(self, processed_experience, modification, sample_round):
        """
        Check if a proposed modification should be allowed.
        
        Key improvements:
        1. Block exact duplicates of failed modifications
        2. Block modifications that contain known harmful patterns (RAG, ToolAgent)
        3. Allow variations of neutral/successful approaches
        """
        modification_lower = modification.lower()
        
        # HARD BLOCK: Known harmful patterns (experimentally proven to decrease score)
        # Note: ToolAgent is now allowed for code execution
        FORBIDDEN_PATTERNS = [
            "ragretriever", "rag_retriever", "rag retriever",
            "add rag", "增加rag", "使用rag",
        ]
        
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in modification_lower:
                logger.info(f"BLOCKED: Modification contains forbidden pattern '{pattern}'")
                return False
        
        experience_data = processed_experience.get(sample_round)
        if experience_data:
            # Block exact duplicates of failures
            for key, value in experience_data["failure"].items():
                if value["modification"] == modification:
                    logger.info(f"BLOCKED: Exact duplicate of failed modification")
                    return False
            # Block exact duplicates of successes (encourage exploration)
            for key, value in experience_data["success"].items():
                if value["modification"] == modification:
                    logger.info(f"BLOCKED: Exact duplicate of successful modification (try variation)")
                    return False
            return True
        else:
            return True  # 如果 experience_data 为空，也返回 True

    def create_experience_data(self, sample, modification):
        return {
            "father node": sample["round"],
            "modification": modification,
            "before": sample["score"],
            "after": None,
            "succeed": None,
        }

    def update_experience(self, directory, experience, avg_score):
        experience["after"] = avg_score
        
        # Add significance threshold to avoid noise-based misjudgment
        # Only consider it a success if improvement > 0.02 (2%)
        # Only consider it a failure if degradation > 0.02 (2%)
        SIGNIFICANCE_THRESHOLD = 0.02
        
        improvement = avg_score - experience["before"]
        
        if improvement > SIGNIFICANCE_THRESHOLD:
            experience["succeed"] = True
            logger.info(f"Experience: SUCCESS (improvement: {improvement:.4f} > threshold {SIGNIFICANCE_THRESHOLD})")
        elif improvement < -SIGNIFICANCE_THRESHOLD:
            experience["succeed"] = False
            logger.info(f"Experience: FAILURE (degradation: {improvement:.4f} < -threshold {SIGNIFICANCE_THRESHOLD})")
        else:
            # Within noise range - mark as neutral (not failure)
            # This prevents good modifications from being blacklisted due to noise
            experience["succeed"] = None  # Neutral - neither success nor failure
            logger.info(f"Experience: NEUTRAL (change: {improvement:.4f} within noise threshold ±{SIGNIFICANCE_THRESHOLD})")

        write_json_file(os.path.join(directory, "experience.json"), experience, encoding="utf-8", indent=4)
