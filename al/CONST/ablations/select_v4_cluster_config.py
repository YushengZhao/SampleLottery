# al/CONST/ablations/select_v4_cluster_config.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
import os
import json
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


def main():
    parser = argparse.ArgumentParser(description="Ablation V4: Cluster with a flexible configuration (k clusters, top n per cluster).")
    parser.add_argument("--num_clusters_v4", type=int, default=4, help="The number of clusters to partition problems into.")
    parser.add_argument("--top_n_per_cluster_v4", type=int, default=2, help="Number of top samples to select from each cluster.")
    parser.add_argument("--pad_to_size", type=int, default=64, help="The final size of the training set after padding.")
    args = parser.parse_args()

    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    model_output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    try:
        embeddings_path = os.path.join(model_output_dir, "problem_embeddings.npz")
        with np.load(embeddings_path, allow_pickle=True) as data:
            df_embeddings = pd.DataFrame({'problem': data['problems'], 'embedding': list(data['embeddings'])})

        screened_results_path = os.path.join(model_output_dir, 'screened_results.jsonl')
        df_uncertainty = pd.read_json(screened_results_path, lines=True).drop_duplicates(subset=['problem'], keep='first')

        with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
            ground_truth_map = {json.loads(line)['problem']: json.loads(line)['ground_truth_answer'] for line in f}
    except FileNotFoundError as e:
        print(f"Error: A required input file was not found. Details: {e}", file=sys.stderr)
        sys.exit(1)

    df_merged = pd.merge(df_uncertainty, df_embeddings, on='problem')
    if len(df_merged) < args.num_clusters_v4:
        print(f"Error: Number of problems ({len(df_merged)}) is less than requested clusters ({args.num_clusters_v4}).", file=sys.stderr)
        sys.exit(1)

    X = np.vstack(df_merged['embedding'].values)
    kmeans = KMeans(n_clusters=args.num_clusters_v4, random_state=42, n_init='auto')
    df_merged['cluster'] = kmeans.fit_predict(X)

    selected_samples_df = df_merged.sort_values(
        by=['cluster', 'al_score'], ascending=[True, False]
    ).groupby('cluster').head(args.top_n_per_cluster_v4)

    selected_list = selected_samples_df.to_dict('records')
    if not selected_list:
        print("Error: No samples were selected. Aborting.", file=sys.stderr)
        sys.exit(1)

    total_selected = len(selected_list)
    num_repeats = args.pad_to_size // total_selected
    num_remainder = args.pad_to_size % total_selected
    final_data_list = (selected_list * num_repeats) + selected_list[:num_remainder]

    data_source_name = "CONST_V4_Cluster4_Top2"

    training_samples = []
    for i, raw_sample in enumerate(tqdm(final_data_list, desc="Formatting final samples")):
        training_samples.append(create_training_sample(
            problem=raw_sample['problem'], ground_truth=ground_truth_map.get(raw_sample['problem']),
            index=i, instruction_prompt=al_cfg.instruction_prompt, data_source_name=data_source_name
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
