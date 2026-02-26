# al/CONST/ablations/select_v6_procedural_only.py
import json
import os
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from tqdm import tqdm
import sys

from al.config import ALConfig


def create_training_sample(problem: str, ground_truth: str, index: int, instruction_prompt: str, data_source_name: str) -> dict:
    """Formats a data row into the final training sample structure."""
    prompt_content = str(problem) + instruction_prompt
    prompt_field = [{'role': 'user', 'content': prompt_content}]
    reward_model_field = {'ground_truth': str(ground_truth), 'style': 'rule'}
    extra_info_field = {'index': index, 'split': 'train'}
    return {"prompt": prompt_field, "reward_model": reward_model_field, "data_source": data_source_name, "ability": "math", "extra_info": extra_info_field}


def calculate_nonconformity(answers: list, true_answer: str, lambda_entropy: float) -> float:
    """Calculates nonconformity score for recalibration."""
    if not answers:
        return float('inf')
    answers = [str(a) for a in answers if a]
    if not answers:
        return float('inf')
    freq = answers.count(str(true_answer)) / len(answers)
    _, counts = np.unique(answers, return_counts=True)
    norm_entropy = entropy(counts, base=len(answers)) if len(counts) > 1 else 0
    return -freq + lambda_entropy * norm_entropy


def calculate_al_score_and_entropy(answers: list, q_hat: float, lambda_entropy: float) -> tuple[int, float]:
    """Calculates AL score and entropy for rescoring the pool."""
    if not answers:
        return 0, 0.0
    answers = [str(a) for a in answers if a]
    if not answers:
        return 0, 0.0

    _, counts = np.unique(answers, return_counts=True)
    calculated_entropy = entropy(counts, base=len(answers)) if len(counts) > 1 else 0.0

    score = sum(1 for y_prime in np.unique(answers) if (-answers.count(y_prime) / len(answers) + lambda_entropy * calculated_entropy) <= q_hat)
    return score, calculated_entropy


def main():
    parser = argparse.ArgumentParser(description="Ablation V6: Use Procedural Volatility only for selection.")
    parser.add_argument("-k", "--num_samples", type=int, default=8, help="Number of samples to select.")
    parser.add_argument("--pad_to_size", type=int, default=64, help="Final size of the training set.")
    args = parser.parse_args()

    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    model_output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Recalculate q_hat using only Procedural Volatility answers
    try:
        with open(al_cfg.calibration_set_path, 'r', encoding='utf-8') as f:
            calib_set = [json.loads(line) for line in f]
        calib_gt_map = {item['problem']: item['ground_truth_answer'] for item in calib_set}

        calib_results_path = os.path.join(model_output_dir, "calibration_results_procedural_volatility.jsonl")
        with open(calib_results_path, 'r', encoding='utf-8') as f:
            calib_answers = {k: v for line in f for k, v in json.loads(line).items()}

        # Also load the ground truth map for the entire pool for the final step
        with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
            full_pool_gt_map = {json.loads(line)['problem']: json.loads(line)['ground_truth_answer'] for line in f}

    except FileNotFoundError as e:
        print(f"Error: Missing a required input file: {e}", file=sys.stderr)
        sys.exit(1)

    scores = [calculate_nonconformity(calib_answers.get(item['problem'], []), calib_gt_map[item['problem']], al_cfg.nonconformity_lambda_entropy) for item in calib_set]
    valid_scores = [s for s in scores if s != float('inf')]

    if not valid_scores:
        print("Error: Could not calculate any valid nonconformity scores for recalibration.", file=sys.stderr)
        sys.exit(1)

    q_hat_level = np.ceil((len(valid_scores) + 1) * (1 - al_cfg.alpha)) / len(valid_scores)
    q_hat = np.quantile(valid_scores, q_hat_level, method="higher")

    # Rescore the pool using the new q_hat
    screened_results_path = os.path.join(model_output_dir, 'screened_results.jsonl')
    df_pool = pd.read_json(screened_results_path, lines=True)

    num_outcome_samples = al_cfg.rollout_overrides.get('n', 20)

    tqdm.pandas(desc="Rescoring Pool (Procedural Only)")
    scores = df_pool['generated_answers'].progress_apply(
        lambda ans: calculate_al_score_and_entropy(ans[num_outcome_samples:], q_hat, al_cfg.nonconformity_lambda_entropy)
    )
    df_pool[['al_score', 'entropy']] = pd.DataFrame(scores.tolist(), index=df_pool.index)

    # Cluster and Select Samples
    embeddings_path = os.path.join(model_output_dir, "problem_embeddings.npz")
    with np.load(embeddings_path, allow_pickle=True) as data:
        df_embeddings = pd.DataFrame({'problem': data['problems'], 'embedding': list(data['embeddings'])})

    df_merged = pd.merge(df_pool, df_embeddings, on='problem')

    X = np.vstack(df_merged['embedding'].values)
    kmeans = KMeans(n_clusters=args.num_samples, random_state=42, n_init='auto')
    df_merged['cluster'] = kmeans.fit_predict(X)

    selected_samples_df = df_merged.sort_values(by=['cluster', 'al_score'], ascending=[True, False]).groupby('cluster').head(1)

    # Build and Save Training Set
    selected_list = selected_samples_df.to_dict('records')
    if not selected_list:
        print("Error: No samples were selected after clustering. Aborting.", file=sys.stderr)
        sys.exit(1)

    num_repeats = args.pad_to_size // len(selected_list)
    num_remainder = args.pad_to_size % len(selected_list)
    final_data_list = (selected_list * num_repeats) + selected_list[:num_remainder]

    # Use a fixed data source name
    data_source_name = "CONST_V6_ProceduralOnly"

    training_samples = []
    for i, raw_sample in enumerate(tqdm(final_data_list, desc="Formatting final samples")):
        training_samples.append(create_training_sample(
            problem=raw_sample['problem'],
            ground_truth=full_pool_gt_map.get(raw_sample['problem']),
            index=i,
            instruction_prompt=al_cfg.instruction_prompt,
            data_source_name=data_source_name
        ))

    if training_samples:
        # Save to the specified AblationStudy directory
        output_dir = os.path.join("data/train", model_name, "AblationStudy")
        os.makedirs(output_dir, exist_ok=True)

        # Use the fixed filename
        output_filename = os.path.join(output_dir, f"{data_source_name}.parquet")

        df_final = pd.DataFrame(training_samples)
        df_final = df_final[['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']]
        df_final.to_parquet(output_filename, index=False)

        # Final confirmation message
        print(f"Task Complete. Final training set saved to: {output_filename}")
    else:
        print("Warning: Could not create any valid training samples.", file=sys.stderr)


if __name__ == "__main__":
    main()
