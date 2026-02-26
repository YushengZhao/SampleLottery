# al/02_screen_pool.py
import json
import os
import sys
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
import ray
from typing import List, Generator

from config import ALConfig
from utils import VLLMSamplingActor


def calculate_entropy(generated_answers: list) -> float:
    """Calculates normalized entropy for a list of answers."""
    if not isinstance(generated_answers, list) or not generated_answers:
        return 0.0
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return 0.0
    _, counts = np.unique(generated_answers_str, return_counts=True)
    return entropy(counts, base=len(generated_answers_str)) if len(counts) > 1 else 0.0


def calculate_prediction_set_size(generated_answers: list, q_hat: float, lambda_entropy: float, precomputed_entropy: float) -> int:
    """Calculates the prediction set size (our AL score)."""
    if not isinstance(generated_answers, list) or not generated_answers:
        return 0
    generated_answers_str = [str(ans) for ans in generated_answers if ans]
    if not generated_answers_str:
        return 0

    prediction_set_size = 0
    for y_prime in np.unique(generated_answers_str):
        freq = generated_answers_str.count(y_prime) / len(generated_answers_str)
        nonconformity_score = -freq + lambda_entropy * precomputed_entropy
        if nonconformity_score <= q_hat:
            prediction_set_size += 1
    return prediction_set_size


def read_prompts_in_chunks(pool_path: str, processed_prompts: set, chunk_size: int) -> Generator[List[str], None, None]:
    """Streams prompts from the pool file, skipping those already processed."""
    chunk = []
    with open(pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                problem = json.loads(line)['problem']
                if problem not in processed_prompts:
                    chunk.append(problem)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            except (json.JSONDecodeError, KeyError):
                continue
    if chunk:
        yield chunk


def main():
    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting Data Pool Screening ---")
    print(f"Model: {model_name}")

    screened_results_path = os.path.join(output_dir, "screened_results.jsonl")
    processed_prompts = set()
    if os.path.exists(screened_results_path):
        with open(screened_results_path, 'r', encoding='utf-8') as f:
            processed_prompts.update(json.loads(line)['problem'] for line in f if line.strip())
        print(f"Found {len(processed_prompts)} existing screened results. Resuming...")

    try:
        with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
            total_in_pool = sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: Data pool file not found at {al_cfg.full_data_pool_path}", file=sys.stderr)
        sys.exit(1)

    if len(processed_prompts) >= total_in_pool:
        print(f"Screening is already complete ({len(processed_prompts)}/{total_in_pool}). Exiting.")
        return

    vllm_actor = None
    try:
        ray.init(address='auto' if os.environ.get("RAY_ADDRESS") else None, ignore_reinit_error=True)

        q_hat_path = os.path.join(output_dir, "q_hat.json")
        try:
            with open(q_hat_path, 'r') as f:
                q_hat = json.load(f)['q_hat']
        except FileNotFoundError:
            print(f"Error: q_hat.json not found at {q_hat_path}. Please run 01_calibrate.py first.", file=sys.stderr)
            sys.exit(1)

        print(f"Using q_hat = {q_hat:.4f} for screening.")

        num_gpus_for_vllm = int(ray.cluster_resources().get("GPU", 1))
        rollout_config_for_actor = al_cfg.rollout_overrides
        rollout_config_for_actor['tensor_model_parallel_size'] = num_gpus_for_vllm

        vllm_actor = VLLMSamplingActor.remote(
            model_path=al_cfg.base_model_path, rollout_config=rollout_config_for_actor,
            instruction_prompt=al_cfg.instruction_prompt, answer_extraction_prompt=al_cfg.answer_extraction_prompt
        )
        print("Initializing vLLM engine for screening...")
        ray.get(vllm_actor.initialize_vllm_engine.remote())
        print("vLLM engine initialized.")

        active_futures, pending_results, batch_id_counter = {}, {}, 0
        with open(screened_results_path, 'a', encoding='utf-8') as f_out, \
                tqdm(total=total_in_pool, initial=len(processed_prompts), desc="Screening Pool") as pbar:

            prompt_generator = read_prompts_in_chunks(al_cfg.full_data_pool_path, processed_prompts, al_cfg.inference_params['screening_batch_size'])
            can_submit_more = True

            while can_submit_more or active_futures:
                while len(active_futures) < al_cfg.inference_params['max_in_flight_tasks'] * 2 and can_submit_more:
                    try:
                        prompt_batch = next(prompt_generator)
                        batch_id = batch_id_counter
                        batch_id_counter += 1
                        pending_results[batch_id] = {"batch": prompt_batch}

                        outcome_future = vllm_actor.get_outcome_volatility_answers.remote(prompt_batch)
                        procedural_future = vllm_actor.get_procedural_volatility_answers.remote(prompt_batch, al_cfg.trajectory_num_checkpoints)

                        active_futures[outcome_future] = {"id": batch_id, "type": "outcome"}
                        active_futures[procedural_future] = {"id": batch_id, "type": "procedural"}
                    except StopIteration:
                        can_submit_more = False

                if not active_futures:
                    break

                done_futures, _ = ray.wait(list(active_futures.keys()), num_returns=1)
                for future in done_futures:
                    task_info = active_futures.pop(future)
                    batch_id, task_type = task_info["id"], task_info["type"]
                    pending_results[batch_id][task_type] = ray.get(future)

                    if "outcome" in pending_results[batch_id] and "procedural" in pending_results[batch_id]:
                        batch_info = pending_results.pop(batch_id)
                        for prompt in batch_info["batch"]:
                            combined = batch_info["outcome"].get(prompt, []) + batch_info["procedural"].get(prompt, [])
                            entropy_score = calculate_entropy(combined)
                            al_score = calculate_prediction_set_size(combined, q_hat, al_cfg.nonconformity_lambda_entropy, entropy_score)
                            f_out.write(json.dumps({"problem": prompt, "al_score": al_score, "entropy": entropy_score, "generated_answers": combined}) + '\n')
                        f_out.flush()
                        pbar.update(len(batch_info["batch"]))

        print(f"Screening complete! Results saved to {screened_results_path}")

    except Exception as e:
        print(f"\nA fatal error occurred during the screening process: {e}", file=sys.stderr)

    finally:
        if vllm_actor:
            print("Shutting down Ray Actor...")
            ray.kill(vllm_actor)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
