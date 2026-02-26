# al/01_calibrate.py
import json
import os
import sys
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import ray
from typing import List, Dict

from config import ALConfig
from utils import VLLMSamplingActor


def calculate_nonconformity(generated_answers: List[str], true_answer: str, lambda_entropy: float) -> float:
    """Calculates the non-conformity score for a set of answers."""
    if not generated_answers:
        return float('inf')

    true_answer_str = str(true_answer)
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return float('inf')

    freq = generated_answers_str.count(true_answer_str) / len(generated_answers_str)
    _, counts = np.unique(generated_answers_str, return_counts=True)
    norm_entropy = entropy(counts, base=len(generated_answers_str)) if len(counts) > 1 else 0

    return -freq + lambda_entropy * norm_entropy


def run_inference_with_resume(actor: ray.actor.ActorHandle, method_name: str, prompts: List[str], result_file_path: str, desc: str, batch_size: int, **kwargs) -> Dict:
    """A general-purpose inference function with resume support."""
    all_results = {}
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_results.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Resumed {len(all_results)} completed '{desc}' results.")

    prompts_to_process = [p for p in prompts if p not in all_results]
    if not prompts_to_process:
        print(f"All '{desc}' samples have already been processed.")
    else:
        prompt_batches = [prompts_to_process[i:i + batch_size] for i in range(0, len(prompts_to_process), batch_size)]
        futures = [getattr(actor, method_name).remote(batch, **kwargs) for batch in prompt_batches]

        with open(result_file_path, 'a', encoding='utf-8') as f_out:
            with tqdm(total=len(prompts_to_process), desc=desc) as pbar:
                while futures:
                    done_ids, futures = ray.wait(futures)
                    for result_dict in ray.get(done_ids):
                        f_out.write(json.dumps(result_dict) + '\n')
                        f_out.flush()
                        all_results.update(result_dict)
                        pbar.update(len(result_dict))
    return all_results


def main():
    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print("--- Starting Calibration Step ---")
    print(f"Model: {model_name}")

    ray.init(address='auto' if os.environ.get("RAY_ADDRESS") else None, ignore_reinit_error=True)

    print(f"Loading calibration set from: {al_cfg.calibration_set_path}")
    try:
        with open(al_cfg.calibration_set_path, 'r', encoding='utf-8') as f:
            calibration_set = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Calibration set file not found at {al_cfg.calibration_set_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(calibration_set)} calibration samples.")

    num_gpus_for_vllm = int(ray.cluster_resources().get("GPU", 1))
    # Directly use and modify the config from ALConfig
    rollout_config_for_actor = al_cfg.rollout_overrides
    rollout_config_for_actor['tensor_model_parallel_size'] = num_gpus_for_vllm
    print(f"Configuring vLLM to use {num_gpus_for_vllm} GPU(s).")

    vllm_actor = VLLMSamplingActor.remote(
        model_path=al_cfg.base_model_path,
        rollout_config=rollout_config_for_actor,
        instruction_prompt=al_cfg.instruction_prompt,
        answer_extraction_prompt=al_cfg.answer_extraction_prompt
    )
    print("Initializing vLLM engine...")
    ray.get(vllm_actor.initialize_vllm_engine.remote())
    print("vLLM engine initialized successfully.")

    all_prompts = [item['problem'] for item in calibration_set]

    outcome_volatility_results = run_inference_with_resume(
        actor=vllm_actor, method_name='get_outcome_volatility_answers', prompts=all_prompts,
        result_file_path=os.path.join(output_dir, "calibration_results_outcome_volatility.jsonl"),
        desc="Calibrate: Outcome Volatility", batch_size=al_cfg.inference_params['calibration_batch_size']
    )

    procedural_volatility_results = run_inference_with_resume(
        actor=vllm_actor, method_name='get_procedural_volatility_answers', prompts=all_prompts,
        result_file_path=os.path.join(output_dir, "calibration_results_procedural_volatility.jsonl"),
        desc="Calibrate: Procedural Volatility", batch_size=al_cfg.inference_params['calibration_batch_size'],
        num_checkpoints=al_cfg.trajectory_num_checkpoints
    )

    ground_truths = {item['problem']: item['ground_truth_answer'] for item in calibration_set}
    nonconformity_scores = []

    print("Calculating fused non-conformity scores...")
    for item in tqdm(calibration_set, desc="Fusing and Scoring"):
        prompt, true_answer = item['problem'], ground_truths.get(item['problem'])
        if true_answer is not None:
            combined_answers = outcome_volatility_results.get(prompt, []) + procedural_volatility_results.get(prompt, [])
            score = calculate_nonconformity(combined_answers, true_answer, lambda_entropy=al_cfg.nonconformity_lambda_entropy)
            if score != float('inf'):
                nonconformity_scores.append(score)

    if not nonconformity_scores:
        print("Error: Failed to calculate any valid non-conformity scores. Calibration cannot proceed.", file=sys.stderr)
        sys.exit(1)

    q_hat_level = np.ceil((len(nonconformity_scores) + 1) * (1 - al_cfg.alpha)) / len(nonconformity_scores)
    q_hat = np.quantile(nonconformity_scores, q_hat_level, method="higher")

    print(f"Calibration complete! Based on {len(nonconformity_scores)} valid scores, q_hat = {q_hat}")
    q_hat_path = os.path.join(output_dir, "q_hat.json")
    with open(q_hat_path, 'w') as f:
        json.dump({"q_hat": q_hat, "num_scores": len(nonconformity_scores)}, f)
    print(f"q_hat saved to {q_hat_path}")

    print("Shutting down Ray Actor...")
    ray.kill(vllm_actor)
    print("Actor shutdown complete.")


if __name__ == "__main__":
    main()
