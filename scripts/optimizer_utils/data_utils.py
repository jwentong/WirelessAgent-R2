import datetime
import json
import os
import random

import numpy as np
import pandas as pd

from scripts.logs import logger
from scripts.utils.common import read_json_file, write_json_file


class DataUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.top_scores = []

    def load_results(self, path: str) -> list:
        result_path = os.path.join(path, "results.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as json_file:
                try:
                    return json.load(json_file)
                except json.JSONDecodeError:
                    return []
        return []

    def get_top_rounds(self, sample: int, path=None, mode="Graph"):
        self._load_scores(path, mode)
        unique_rounds = set()
        unique_top_scores = []

        first_round = next((item for item in self.top_scores if item["round"] == 1), None)
        if first_round:
            unique_top_scores.append(first_round)
            unique_rounds.add(1)

        for item in self.top_scores:
            if item["round"] not in unique_rounds:
                unique_top_scores.append(item)
                unique_rounds.add(item["round"])

                if len(unique_top_scores) >= sample:
                    break

        return unique_top_scores

    def select_round(self, items, experience_data=None):
        """
        Select a round using modified UCB-style selection.
        
        Key improvement: Penalize nodes that have been tried many times with failures.
        
        Args:
            items: List of {round, score} dictionaries
            experience_data: Optional processed_experience dict to adjust probabilities
        """
        if not items:
            raise ValueError("Item list is empty.")

        sorted_items = sorted(items, key=lambda x: x["score"], reverse=True)
        scores = [item["score"] * 100 for item in sorted_items]
        
        # Calculate penalty based on exploration history
        penalties = self._calculate_exploration_penalties(sorted_items, experience_data)

        probabilities = self._compute_probabilities(scores, penalties=penalties)
        logger.info(f"\nMixed probability distribution: {probabilities}")
        logger.info(f"\nSorted rounds: {sorted_items}")
        if experience_data:
            logger.info(f"\nExploration penalties: {penalties}")

        selected_index = np.random.choice(len(sorted_items), p=probabilities)
        logger.info(f"\nSelected index: {selected_index}, Selected item: {sorted_items[selected_index]}")

        return sorted_items[selected_index]

    def _calculate_exploration_penalties(self, items, experience_data):
        """
        Calculate penalty for each round based on exploration history.
        
        Intuition:
        - If a round has many failed children, it's "saturated" - penalize it
        - If a round has no children, it's unexplored - no penalty
        - If a round has successful children, slightly reduce penalty
        
        Returns: List of penalty multipliers (0.0-1.0, lower = more penalty)
        """
        if not experience_data:
            return [1.0] * len(items)
        
        penalties = []
        for item in items:
            round_num = str(item["round"])
            exp = experience_data.get(round_num, {})
            
            n_success = len(exp.get("success", {}))
            n_failure = len(exp.get("failure", {}))
            n_neutral = len(exp.get("neutral", {}))
            n_total = n_success + n_failure + n_neutral
            
            if n_total == 0:
                # Unexplored node - no penalty
                penalty = 1.0
            else:
                # Penalty based on failure ratio
                # If all children failed, penalty = 0.3 (heavily penalized)
                # If all children succeeded, penalty = 1.2 (boost!)
                failure_ratio = n_failure / n_total
                success_ratio = n_success / n_total
                
                # Base penalty: reduce by failure ratio
                penalty = 1.0 - 0.7 * failure_ratio + 0.2 * success_ratio
                
                # Additional penalty if many attempts made (node is "saturated")
                if n_total >= 3:
                    penalty *= 0.8  # Extra penalty for heavily explored nodes
                    
                penalty = max(0.1, min(1.3, penalty))  # Clamp to [0.1, 1.3]
            
            penalties.append(penalty)
            
        return penalties

    def _compute_probabilities(self, scores, alpha=0.2, lambda_=0.3, penalties=None):
        scores = np.array(scores, dtype=np.float64)
        n = len(scores)

        if n == 0:
            raise ValueError("Score list is empty.")

        uniform_prob = np.full(n, 1.0 / n, dtype=np.float64)

        max_score = np.max(scores)
        shifted_scores = scores - max_score
        exp_weights = np.exp(alpha * shifted_scores)
        
        # Apply exploration penalties
        if penalties is not None:
            penalties = np.array(penalties, dtype=np.float64)
            exp_weights = exp_weights * penalties

        sum_exp_weights = np.sum(exp_weights)
        if sum_exp_weights == 0:
            raise ValueError("Sum of exponential weights is 0, cannot normalize.")

        score_prob = exp_weights / sum_exp_weights

        mixed_prob = lambda_ * uniform_prob + (1 - lambda_) * score_prob

        total_prob = np.sum(mixed_prob)
        if not np.isclose(total_prob, 1.0):
            mixed_prob = mixed_prob / total_prob

        return mixed_prob

    def load_log(self, cur_round, path=None, mode: str = "Graph"):
        if mode == "Graph":
            log_dir = os.path.join(self.root_path, "workflows", f"round_{cur_round}", "log.json")
        else:
            log_dir = path

        # 检查文件是否存在
        if not os.path.exists(log_dir):
            return ""  # 如果文件不存在，返回空字符串
        logger.info(log_dir)
        data = read_json_file(log_dir, encoding="utf-8")

        if isinstance(data, dict):
            data = [data]
        elif not isinstance(data, list):
            data = list(data)

        if not data:
            return ""

        sample_size = min(3, len(data))
        random_samples = random.sample(data, sample_size)

        log = ""
        for sample in random_samples:
            log += json.dumps(sample, indent=4, ensure_ascii=False) + "\n\n"

        return log

    def get_results_file_path(self, graph_path: str) -> str:
        return os.path.join(graph_path, "results.json")

    def create_result_data(self, round: int, score: float, avg_cost: float, total_cost: float) -> dict:
        now = datetime.datetime.now()
        return {"round": round, "score": score, "avg_cost": avg_cost, "total_cost": total_cost, "time": now}

    def save_results(self, json_file_path: str, data: list):
        write_json_file(json_file_path, data, encoding="utf-8", indent=4)

    def _load_scores(self, path=None, mode="Graph"):
        if mode == "Graph":
            rounds_dir = os.path.join(self.root_path, "workflows")
        else:
            rounds_dir = path

        result_file = os.path.join(rounds_dir, "results.json")
        self.top_scores = []

        data = read_json_file(result_file, encoding="utf-8")
        df = pd.DataFrame(data)

        scores_per_round = df.groupby("round")["score"].mean().to_dict()

        for round_number, average_score in scores_per_round.items():
            self.top_scores.append({"round": round_number, "score": average_score})

        self.top_scores.sort(key=lambda x: x["score"], reverse=True)

        return self.top_scores
