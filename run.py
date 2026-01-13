# -*- coding: utf-8 -*-
# @Date    : 1/13/2026
# @Author  : Jingwen
# @Desc    : Entrance of WirelessAgent.

import argparse
import os
import sys
from typing import Dict, List

# Fix Windows console encoding issues
if sys.platform == 'win32':
    # Set environment variable for Python to use UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        # Reconfigure stdout/stderr to use UTF-8 encoding
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        # For older Python versions
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass

from data.download_data import download
from scripts.optimizer import Optimizer
from scripts.async_llm import LLMsConfig

class ExperimentConfig:
    def __init__(self, dataset: str, question_type: str, operators: List[str]):
        self.dataset = dataset
        self.question_type = question_type
        self.operators = operators


EXPERIMENT_CONFIGS: Dict[str, ExperimentConfig] = {
    "WCHW": ExperimentConfig(
        dataset="WCHW",
        question_type="math",
        operators=["Custom", "ScEnsemble", "ToolAgent"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="WirelessAgent Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--optimized_path",
        type=str,
        default="workspace",
        help="Optimized result save path",
    )
    parser.add_argument("--initial_round", type=int, default=1, help="Initial round")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max iteration rounds")
    parser.add_argument("--check_convergence", type=bool, default=True, help="Whether to enable early stop")
    parser.add_argument("--validation_rounds", type=int, default=1, help="Validation rounds")
    parser.add_argument(
        "--if_force_download",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether enforce dataset download.",
    )
    parser.add_argument(
        "--opt_model_name",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Specifies the name of the model used for optimization tasks.",
    )
    parser.add_argument(
        "--exec_model_name",
        type=str,
        default="gpt-4o-mini",
        help="Specifies the name of the model used for execution tasks.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["Graph", "Test"],
        default="Graph",
        help="Running mode: 'Graph' for optimization on validation set, 'Test' for evaluation on test set",
    )
    parser.add_argument(
        "--test_rounds",
        type=str,
        default="1",
        help="Comma-separated round numbers to test (e.g., '1,3,12' or 'all' for all rounds)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = LLMsConfig.default()
    opt_llm_config = models_config.get(args.opt_model_name)
    if opt_llm_config is None:
        raise ValueError(
            f"The optimization model '{args.opt_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --opt_model_name flag. "
        )

    exec_llm_config = models_config.get(args.exec_model_name)
    if exec_llm_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model_name}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model_name flag. "
        )

    download(["datasets"], force_download=args.if_force_download) # remove download initial_rounds in new version.

    optimizer = Optimizer(
        dataset=config.dataset,
        question_type=config.question_type,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
        check_convergence=args.check_convergence,
        operators=config.operators,
        optimized_path=args.optimized_path,
        sample=args.sample,
        initial_round=args.initial_round,
        max_rounds=args.max_rounds,
        validation_rounds=args.validation_rounds,
        test_rounds=args.test_rounds,  # NEW: pass test_rounds parameter
    )

    # Run optimizer in specified mode
    optimizer.optimize(args.mode)

