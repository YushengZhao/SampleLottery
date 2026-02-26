# al/config.py
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ALConfig:
    # --- Core Active Learning Parameters ---
    base_model_path: str = "./llama/LLaMA-3.1-8B-Instruct"
    # base_model_path: str = "./deepseek/DeepSeek-R1-Distill-Qwen-1.5B"
    # base_model_path: str = "./qwen/Qwen2.5-Math-1.5B"

    # --- Data and Output Paths ---
    full_data_pool_path: str = "data/dataset/UnlabeledTrainingPool_BigMath.jsonl"
    calibration_set_path: str = "data/dataset/CalibrationSet_BigMath.jsonl"
    mmlu_calibration_path: str = "data/dataset/CalibrationSet_MMLU.parquet"
    output_dir: str = "al_cache"

    # --- Calibration and Uncertainty Calculation Config ---
    alpha: float = 0.1  # Conformal Prediction error rate (confidence = 1 - alpha)

    # Number of checkpoints for Procedural Volatility exploration
    trajectory_num_checkpoints: int = 20

    # Weight for the normalized entropy term in the non-conformity score (lambda)
    nonconformity_lambda_entropy: float = 0.02

    # --- Inference Performance Parameters ---
    inference_params: Dict = field(default_factory=lambda: {
        "calibration_batch_size": 256,
        "screening_batch_size": 1024,
        "max_in_flight_tasks": 8,
    })

    # --- Prompt Templates ---
    instruction_prompt: str = "\n\nLet's think step by step and output the final answer within \\boxed{}."
    answer_extraction_prompt: str = """You are an expert mathematician and a precise answer extractor. Your task is to analyze the provided mathematical reasoning and extract only the final numerical answer. Do not provide any explanation or preamble. Your final output should ONLY be the answer enclosed in a \\boxed{}.
"""

    # --- Prompt Templates specifically for MMLU Calibration ---
    # These are crucial for handling the multiple-choice format of MMLU.
    mmlu_instruction_prompt: str = """
    You will be presented with a single-choice question. Please analyze the question and the provided options to determine the single correct answer.

    Your final response should be ONLY the letter of the correct option (e.g., A, B, C, or D) enclosed in a \\boxed{}. For example, if the correct option is B, your response must be \\boxed{B}.
    """
    mmlu_answer_extraction_prompt: str = """You are a precise answer extractor. Your task is to analyze the provided reasoning for a single-choice question and determine the correct option.

    Your final output must be ONLY the letter of the correct option (e.g., A, B, C, or D) enclosed in a \\boxed{}.
    """

    # --- Rollout Configuration Overrides ---
    # This config is essential for controlling the vLLM sampling behavior.
    rollout_overrides: dict = field(default_factory=lambda: {
        "name": "vllm",
        "distributed_executor_backend": "ray",
        "tensor_model_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "temperature": 0.6,
        "n": 20,  # Controls the number of samples for Outcome Volatility
        "dtype": "bfloat16",
        "prompt_length": 1024,
        "response_length": 2048,
        "max_model_len": 4096,
        "max_prompt_length": 1024,
        "max_response_length": 2048,
        "enforce_eager": True,
        "free_cache_engine": False,
    })
