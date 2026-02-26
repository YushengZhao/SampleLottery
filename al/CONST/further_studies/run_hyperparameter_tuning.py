import json
import os
import argparse
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import entropy
from tqdm import tqdm

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
    clean_answers = [str(a) for a in answers if a]
    if not clean_answers:
        return float('inf')
    freq = clean_answers.count(str(true_answer)) / len(clean_answers)
    _, counts = np.unique(clean_answers, return_counts=True)
    norm_entropy = entropy(counts, base=len(clean_answers)) if len(counts) > 1 else 0
    return -freq + lambda_entropy * norm_entropy


def calculate_al_score_and_entropy(answers: list, q_hat: float, lambda_entropy: float) -> tuple[int, float]:
    """Calculates AL score and entropy for rescoring the pool."""
    if not answers:
        return 0, 0.0
    clean_answers = [str(a) for a in answers if a]
    if not clean_answers:
        return 0, 0.0
    _, counts = np.unique(clean_answers, return_counts=True)
    calculated_entropy = entropy(counts, base=len(clean_answers)) if len(counts) > 1 else 0.0
    score = sum(1 for y in np.unique(clean_answers) if (-clean_answers.count(y) / len(clean_answers) + lambda_entropy * calculated_entropy) <= q_hat)
    return score, calculated_entropy


def get_proportional_slice(full_list: list, num_to_select: int) -> list:
    """Selects a proportional subset of items from a list."""
    total_available = len(full_list)
    if num_to_select >= total_available:
        return full_list
    if num_to_select == 0:
        return []
    indices = np.linspace(0, total_available - 1, num_to_select, dtype=int)
    return [full_list[i] for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Further Study: Tune hyperparameters for outcome and procedural volatility samples.")
    parser.add_argument("--num_outcome", type=int, default=20, help="Number of outcome volatility samples to use.")
    parser.add_argument("--num_procedural", type=int, default=20, help="Number of procedural volatility samples to use.")
    parser.add_argument("-k", "--num_samples", type=int, default=8, help="Final number of samples to select after clustering.")
    parser.add_argument("--pad_to_size", type=int, default=64, help="Final size of the training set.")
    args = parser.parse_args()

    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    model_output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"--- Starting Hyperparameter Tuning ---")
    print(f"Using {args.num_outcome} outcome samples and {args.num_procedural} procedural samples.")

    # Recalibrate q_hat with the specified number of samples
    try:
        with open(al_cfg.calibration_set_path, 'r', encoding='utf-8') as f:
            calib_set = [json.loads(line) for line in f]
        # Load ground truths for both calibration and the final training set creation
        with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
            full_pool_gt_map = {json.loads(line)['problem']: json.loads(line)['ground_truth_answer'] for line in f}

        with open(os.path.join(model_output_dir, "calibration_results_outcome_volatility.jsonl"), 'r', encoding='utf-8') as f:
            outcome_answers = {k: v for line in f for k, v in json.loads(line).items()}
        with open(os.path.join(model_output_dir, "calibration_results_procedural_volatility.jsonl"), 'r', encoding='utf-8') as f:
            procedural_answers = {k: v for line in f for k, v in json.loads(line).items()}
    except FileNotFoundError as e:
        print(f"Error: Missing a required calibration or data file: {e}", file=sys.stderr)
        sys.exit(1)

    scores = []
    for item in tqdm(calib_set, desc="Recalibrating q_hat"):
        outcome_subset = get_proportional_slice(outcome_answers.get(item['problem'], []), args.num_outcome)
        procedural_subset = get_proportional_slice(procedural_answers.get(item['problem'], []), args.num_procedural)
        combined = outcome_subset + procedural_subset
        scores.append(calculate_nonconformity(combined, full_pool_gt_map.get(item['problem']), al_cfg.nonconformity_lambda_entropy))

    valid_scores = [s for s in scores if s != float('inf')]
    if not valid_scores:
        print("Error: Could not calculate any valid nonconformity scores for recalibration.", file=sys.stderr)
        sys.exit(1)

    q_hat_level = np.ceil((len(valid_scores) + 1) * (1 - al_cfg.alpha)) / len(valid_scores)
    q_hat = np.quantile(valid_scores, q_hat_level, method="higher")
    print(f"New q_hat for this configuration: {q_hat:.4f}")

    # Rescore the pool, cluster, and select samples
    df_pool = pd.read_json(os.path.join(model_output_dir, 'screened_results.jsonl'), lines=True)
    TOTAL_OUTCOME_SAMPLES = al_cfg.rollout_overrides.get('n', 20)

    tqdm.pandas(desc="Rescoring Pool")

    def rescore_row(row):
        full_answers = row['generated_answers']
        outcome_subset = get_proportional_slice(full_answers[:TOTAL_OUTCOME_SAMPLES], args.num_outcome)
        procedural_subset = get_proportional_slice(full_answers[TOTAL_OUTCOME_SAMPLES:], args.num_procedural)
        combined = outcome_subset + procedural_subset
        return calculate_al_score_and_entropy(combined, q_hat, al_cfg.nonconformity_lambda_entropy)

    df_pool[['al_score', 'entropy']] = df_pool.progress_apply(rescore_row, axis=1, result_type='expand')

    with np.load(os.path.join(model_output_dir, "problem_embeddings.npz"), allow_pickle=True) as data:
        df_embeddings = pd.DataFrame({'problem': data['problems'], 'embedding': list(data['embeddings'])})
    df_merged = pd.merge(df_pool, df_embeddings, on='problem')

    kmeans = KMeans(n_clusters=args.num_samples, random_state=42, n_init='auto')
    df_merged['cluster'] = kmeans.fit_predict(np.vstack(df_merged['embedding'].values))
    selected_samples_df = df_merged.sort_values(by=['cluster', 'al_score'], ascending=False).groupby('cluster').head(1)

    # Build and Save Training Set
    selected_list = selected_samples_df.to_dict('records')
    total_selected = len(selected_list)
    if total_selected == 0:
        print("Error: No samples were selected. Aborting.", file=sys.stderr)
        sys.exit(1)

    num_repeats = args.pad_to_size // total_selected
    num_remainder = args.pad_to_size % total_selected
    final_data_list = (selected_list * num_repeats) + selected_list[:num_remainder]

    data_source_name = f"CONST_Tune_o{args.num_outcome}_p{args.num_procedural}_k{total_selected}"
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
        output_dir = os.path.join("data/train", model_name, "FurtherAnalysis")
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{data_source_name}.parquet")

        df_final = pd.DataFrame(training_samples)
        df_final = df_final[['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']]
        df_final.to_parquet(output_filename, index=False)

        print(f"\n--- Hyperparameter Tuning Run Complete! ---")
        print(f"Training set saved to: {output_filename}")
    else:
        print("Warning: Could not create any valid training samples.", file=sys.stderr)


if __name__ == "__main__":
    main()
