# al/CONST/further_studies/run_robustness_analysis.py
import json
import os
from typing import List, Dict
import numpy as np
import pandas as pd
import ray
from scipy.stats import entropy
from tqdm import tqdm

from al.config import ALConfig
from al.utils import VLLMSamplingActor


def calculate_nonconformity(generated_answers: list[str], true_answer: str, lambda_entropy: float) -> float:
    if not generated_answers:
        return float('inf')
    true_answer_str = str(true_answer)
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return float('inf')
    total_answers = len(generated_answers_str)
    freq = generated_answers_str.count(true_answer_str) / total_answers
    _, counts = np.unique(generated_answers_str, return_counts=True)
    norm_entropy = entropy(counts, base=total_answers) if len(counts) > 1 else 0.0
    return -freq + lambda_entropy * norm_entropy


def calculate_entropy(generated_answers: list) -> float:
    if not isinstance(generated_answers, list) or not generated_answers:
        return 0.0
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return 0.0
    total_answers = len(generated_answers_str)
    _, counts = np.unique(generated_answers_str, return_counts=True)
    return entropy(counts, base=total_answers) if len(counts) > 1 else 0.0


def calculate_al_score(generated_answers: list, q_hat: float, lambda_entropy: float) -> int:
    if not isinstance(generated_answers, list) or not generated_answers:
        return 0
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return 0
    unique_answers = np.unique(generated_answers_str)
    prediction_set_size = 0
    total_samples = len(generated_answers_str)
    prompt_entropy = calculate_entropy(generated_answers_str)
    for y_prime in unique_answers:
        freq = generated_answers_str.count(str(y_prime)) / total_samples
        nonconformity_score = -freq + lambda_entropy * prompt_entropy
        if nonconformity_score <= q_hat:
            prediction_set_size += 1
    return prediction_set_size


def format_mmlu_problem(row: pd.Series) -> str:
    question = row['question']
    choices = row['choices']
    choice_labels = ['A', 'B', 'C', 'D']
    formatted_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(choice_labels, choices)])
    return f"Question: {question}\nChoices:\n{formatted_choices}"


def load_mmlu_calibration_set(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"MMLU Parquet file not found: {path}")
    math_subjects = ['abstract_algebra', 'college_mathematics', 'elementary_mathematics', 'high_school_mathematics', 'high_school_statistics']
    df = pd.read_parquet(path)
    math_df = df[df['subject'].isin(math_subjects)]
    calibration_set = []
    choice_labels = ['A', 'B', 'C', 'D']
    for _, row in math_df.iterrows():
        problem_text = format_mmlu_problem(row)
        ground_truth_answer = choice_labels[row['answer']]
        calibration_set.append({"problem": problem_text, "ground_truth_answer": ground_truth_answer})
    return calibration_set


def load_bigmath_pool_data(pool_path: str) -> List[Dict]:
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Screening pool result file not found: {pool_path}")
    pool_data = []
    with open(pool_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading BIG-MATH pool file"):
            try:
                data = json.loads(line)
                if "generated_answers" in data and "problem" in data:
                    pool_data.append({"problem": data["problem"], "combined_answers": data["generated_answers"]})
            except json.JSONDecodeError:
                continue
    return pool_data


def run_inference_with_resume(actor: ray.actor.ActorHandle, method_name: str, prompts: List[str], result_file_path: str, desc: str, batch_size: int, **kwargs) -> Dict:
    all_results = {}
    if os.path.exists(result_file_path):
        with open(result_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    all_results.update(json.loads(line))
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: Found {len(all_results)} completed results for '{desc}'.")
    processed_prompts = set(all_results.keys())
    prompts_to_process = [p for p in prompts if p not in processed_prompts]
    if not prompts_to_process:
        print(f"All '{desc}' samples have already been processed.")
    else:
        print(f"Processing {len(prompts_to_process)} new '{desc}' samples.")
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
    """Main function to run the calibration and screening pipeline."""
    cfg = ALConfig()
    model_name = os.path.basename(cfg.base_model_path.strip('/'))
    output_dir = os.path.join(cfg.output_dir, model_name, 'further_studies')
    os.makedirs(output_dir, exist_ok=True)
    ray.init(address='auto' if os.environ.get("RAY_ADDRESS") else None, ignore_reinit_error=True)

    # Part 1: Calibration using MMLU
    print("--- Part 1: Starting MMLU Calibration ---")
    print(f"Model: {model_name}")
    num_gpus_for_vllm = int(ray.cluster_resources().get("GPU", 1))
    cfg.rollout_overrides['tensor_model_parallel_size'] = num_gpus_for_vllm
    print(f"Configuring vLLM to use {num_gpus_for_vllm} GPU(s).")
    print("Creating vLLM Actor for MMLU Calibration...")
    vllm_actor = VLLMSamplingActor.remote(
        model_path=cfg.base_model_path,
        rollout_config=cfg.rollout_overrides,
        instruction_prompt=cfg.mmlu_instruction_prompt,
        answer_extraction_prompt=cfg.mmlu_answer_extraction_prompt
    )
    print("Initializing vLLM Engine via remote call...")
    ray.get(vllm_actor.initialize_vllm_engine.remote())
    print("vLLM Engine initialized successfully.")

    calibration_set = load_mmlu_calibration_set(cfg.mmlu_calibration_path)
    all_prompts = [item['problem'] for item in calibration_set]
    instance_results = run_inference_with_resume(
        actor=vllm_actor, method_name='get_outcome_volatility_answers', prompts=all_prompts,
        result_file_path=os.path.join(output_dir, "calibration_mmlu_outcome_volatility.jsonl"),
        desc="Calibrating (MMLU): Outcome Volatility", batch_size=cfg.inference_params["calibration_batch_size"]
    )
    trajectory_results = run_inference_with_resume(
        actor=vllm_actor, method_name='get_procedural_volatility_answers', prompts=all_prompts,
        result_file_path=os.path.join(output_dir, "calibration_mmlu_procedural_volatility.jsonl"),
        desc="Calibrating (MMLU): Procedural Volatility", batch_size=cfg.inference_params["calibration_batch_size"],
        num_checkpoints=cfg.trajectory_num_checkpoints
    )

    ground_truths = {item['problem']: item['ground_truth_answer'] for item in calibration_set}
    nonconformity_scores = []
    print("Calculating fused nonconformity scores for MMLU...")
    for item in tqdm(calibration_set, desc="Fusing scores"):
        prompt, true_answer = item['problem'], ground_truths.get(item['problem'])
        if true_answer is not None:
            combined_answers = instance_results.get(prompt, []) + trajectory_results.get(prompt, [])
            score = calculate_nonconformity(combined_answers, true_answer, cfg.nonconformity_lambda_entropy)
            if score != float('inf'):
                nonconformity_scores.append(score)

    if not nonconformity_scores:
        raise ValueError("Failed to calculate any valid nonconformity scores. Cannot calibrate.")

    q_hat_level = np.ceil((len(nonconformity_scores) + 1) * (1 - cfg.alpha)) / len(nonconformity_scores)
    final_q_hat = np.quantile(nonconformity_scores, q_hat_level, method="higher")

    print("--- Calibration Complete ---")
    print(f"  - Based on {len(nonconformity_scores)} valid MMLU scores.")
    print(f"  - Final q_hat (from MMLU) = {final_q_hat:.4f}")

    # [MODIFIED] Save the MMLU-calibrated q_hat to a distinctly named file.
    q_hat_path = os.path.join(output_dir, "q_hat_mmlu_calibrated.json")
    with open(q_hat_path, 'w') as f:
        json.dump({
            "q_hat": final_q_hat,
            "num_scores": len(nonconformity_scores),
            "calibration_source": "MMLU"
        }, f, indent=4)
    print(f"MMLU-calibrated q_hat saved to: {q_hat_path}")

    # Part 2: Screening the BIG-MATH pool
    print("\n--- Part 2: Screening the BIG-MATH data pool ---")
    pool_results_path = os.path.join(cfg.output_dir, model_name, "screened_results.jsonl")
    pool_data = load_bigmath_pool_data(pool_results_path)
    final_output_path = os.path.join(output_dir, "screened_results_mmlu_calibrated.jsonl")
    print(f"Final MMLU-calibrated results will be saved to: {final_output_path}")

    al_scores = []
    with open(final_output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(pool_data, desc="Calculating AL scores for BIG-MATH with MMLU q_hat"):
            combined_answers = item['combined_answers']
            entropy_score = calculate_entropy(combined_answers)
            al_score = calculate_al_score(combined_answers, final_q_hat, cfg.nonconformity_lambda_entropy)
            al_scores.append(al_score)
            record = {"problem": item["problem"], "al_score": al_score, "entropy": entropy_score, "generated_answers": combined_answers}
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Final Analysis and Cleanup
    print("\n--- Screening Analysis ---")
    al_scores = np.array(al_scores)
    if len(al_scores) > 0:
        zero_count = np.sum(al_scores == 0)
        print(f"Total BIG-MATH samples screened: {len(al_scores)}")
        print(f"Samples with AL score of 0: {zero_count} ({zero_count / len(al_scores):.2%})")
        print(f"Average AL score: {np.mean(al_scores):.2f}")
        print(f"Median AL score: {np.median(al_scores):.0f}")
        print(f"Max AL score: {np.max(al_scores)}")

    print(f"\nScreening complete! Results saved to: {final_output_path}")
    print("Shutting down Ray Actor...")
    ray.kill(vllm_actor)
    ray.shutdown()
    print("Process finished.")


if __name__ == "__main__":
    main()
