from scripts.evaluator import Evaluator


class EvaluationUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_initial_round(self, optimizer, graph_path, directory, validation_n, data):
        # Load graph with graph_utils from optimizer
        optimizer.graph = optimizer.graph_utils.load_graph(optimizer.round, graph_path)
        evaluator = Evaluator(eval_path=directory)

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            new_data = optimizer.data_utils.create_result_data(optimizer.round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(graph_path)
            optimizer.data_utils.save_results(result_path, data)

        return data

    async def evaluate_graph(self, optimizer, directory, validation_n, data, initial=False):
        evaluator = Evaluator(eval_path=directory)
        scores = []  # Collect all scores for robust statistics

        for i in range(validation_n):
            score, avg_cost, total_cost = await evaluator.graph_evaluate(
                optimizer.dataset,
                optimizer.graph,
                {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
                directory,
                is_test=False,
            )

            cur_round = optimizer.round + 1 if initial is False else optimizer.round

            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(f"{optimizer.root_path}/workflows")
            optimizer.data_utils.save_results(result_path, data)

            scores.append(score)

        # Use median instead of mean for robustness against outliers
        scores.sort()
        n = len(scores)
        if n % 2 == 0:
            median_score = (scores[n//2 - 1] + scores[n//2]) / 2
        else:
            median_score = scores[n//2]
        
        from scripts.logs import logger
        logger.info(f"Evaluation scores: {scores}, median: {median_score:.4f}, mean: {sum(scores)/n:.4f}")
        
        return median_score  # Return median for more stable comparison

    async def evaluate_graph_test(self, optimizer, directory, is_test=True):
        evaluator = Evaluator(eval_path=directory)
        return await evaluator.graph_evaluate(
            optimizer.dataset,
            optimizer.graph,
            {"dataset": optimizer.dataset, "llm_config": optimizer.execute_llm_config},
            directory,
            is_test=is_test,
        )
