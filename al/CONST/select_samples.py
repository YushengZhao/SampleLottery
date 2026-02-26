import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
import os
import json
import sys
from tqdm import tqdm

from al.config import ALConfig


def create_training_sample(problem: str, ground_truth: str, index: int, instruction_prompt: str, data_source_name: str) -> dict:
    """A standardized function to format a data row into the final training sample structure."""
    prompt_content = str(problem) + instruction_prompt
    prompt_field = [{'role': 'user', 'content': prompt_content}]
    reward_model_field = {'ground_truth': str(ground_truth), 'style': 'rule'}
    extra_info_field = {'index': index, 'split': 'train'}
    return {
        "prompt": prompt_field,
        "reward_model": reward_model_field,
        "data_source": data_source_name,
        "ability": "math",
        "extra_info": extra_info_field
    }


def main():
    parser = argparse.ArgumentParser(description="Cluster problems, select top samples from each cluster based on AL Score, and build the training set.")
    parser.add_argument("-k", "--num_clusters", type=int, default=8, help="The number of clusters to partition the problems into.")
    parser.add_argument("--top_n_per_cluster", type=int, default=1, help="Number of top samples to select from each cluster, ranked by AL score.")
    parser.add_argument("--pad_to_size", type=int, default=64, help="The final size of the training set after repeating the selected samples.")
    parser.add_argument(
        "--screened_results_file",
        type=str,
        default="screened_results.jsonl",
        help="Filename of the screened results JSONL within the model's AL cache directory."
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="",
        help="Optional subdirectory within 'data/train/<model_name>/' to save the output."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="CONST",
        help="Prefix for the output Parquet file name."
    )
    args = parser.parse_args()

    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))
    model_output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load All Necessary Data
    try:
        embeddings_path = os.path.join(model_output_dir, "problem_embeddings.npz")
        with np.load(embeddings_path, allow_pickle=True) as data:
            df_embeddings = pd.DataFrame({'problem': data['problems'], 'embedding': list(data['embeddings'])})

        screened_results_path = os.path.join(model_output_dir, args.screened_results_file)
        if not os.path.exists(screened_results_path):
            raise FileNotFoundError(f"Screened results file not found: {screened_results_path}")
        df_uncertainty = pd.read_json(screened_results_path, lines=True).drop_duplicates(subset=['problem'], keep='first')

        with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
            ground_truth_map = {json.loads(line)['problem']: json.loads(line)['ground_truth_answer'] for line in f}
    except FileNotFoundError as e:
        print(f"Error: A required input file was not found. Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Merge Data
    df_merged = pd.merge(df_uncertainty, df_embeddings, on='problem')
    if len(df_merged) < args.num_clusters:
        print(f"Error: Number of available problems ({len(df_merged)}) is less than requested clusters ({args.num_clusters}).", file=sys.stderr)
        sys.exit(1)

    # Perform K-Means Clustering
    X = np.vstack(df_merged['embedding'].values)
    kmeans = KMeans(n_clusters=args.num_clusters, random_state=42, n_init='auto')
    df_merged['cluster'] = kmeans.fit_predict(X)

    # Select Top Samples from Each Cluster
    selected_samples_df = df_merged.sort_values(
        by=['cluster', 'al_score'],
        ascending=[True, False]
    ).groupby('cluster').head(args.top_n_per_cluster)

    # Build and Pad the Final Training Set
    selected_samples_list = selected_samples_df.to_dict('records')
    if not selected_samples_list:
        print("Error: No samples were selected. Aborting.", file=sys.stderr)
        sys.exit(1)

    total_selected = len(selected_samples_list)
    num_repeats = args.pad_to_size // total_selected
    num_remainder = args.pad_to_size % total_selected
    final_data_list = (selected_samples_list * num_repeats) + selected_samples_list[:num_remainder]

    data_source_name = f"{args.output_prefix}_k{total_selected}"

    training_samples = []
    for i, raw_sample in enumerate(tqdm(final_data_list, desc=f"Formatting {data_source_name} samples")):
        problem, ground_truth = raw_sample['problem'], ground_truth_map.get(raw_sample['problem'])
        if ground_truth:
            training_samples.append(create_training_sample(
                problem=problem, ground_truth=ground_truth, index=i,
                instruction_prompt=al_cfg.instruction_prompt, data_source_name=data_source_name
            ))

    if training_samples:
        output_dir = os.path.join("data/train", model_name, args.output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        output_filename = os.path.join(output_dir, f"{data_source_name}.parquet")

        df_final = pd.DataFrame(training_samples)
        df_final = df_final[['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']]
        df_final.to_parquet(output_filename, index=False)

        print(f"Task complete. Final training set saved to: {output_filename}")
    else:
        print("Warning: Could not create any valid training samples.", file=sys.stderr)


if __name__ == "__main__":
    main()