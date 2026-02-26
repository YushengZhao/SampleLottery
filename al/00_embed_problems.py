# al/00_embed_problems.py
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm
import argparse
import os
import json
import sys
from typing import List, Dict

from config import ALConfig


def get_embeddings_batched(
        problems: List[str],
        model,
        tokenizer,
        max_length: int = 1024
) -> Dict[str, np.ndarray]:
    """
    Generates embeddings for a batch of problem texts.
    Returns a dictionary mapping each problem to its embedding vector.
    """
    results = {}
    try:
        inputs = tokenizer(problems, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.shape)

        masked_hidden_states = last_hidden_states * attention_mask
        summed_hidden_states = torch.sum(masked_hidden_states, dim=1)
        num_non_padding_tokens = attention_mask.sum(dim=1)

        mean_pooled_embeddings = summed_hidden_states / torch.clamp(num_non_padding_tokens, min=1e-9)

        cpu_embeddings = mean_pooled_embeddings.cpu().numpy()
        for i, problem_text in enumerate(problems):
            results[problem_text] = cpu_embeddings[i]
    except Exception:
        # Return an empty dict if the batch fails, letting the main loop handle it.
        return {}

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate and cache embeddings for all problems in the data pool.")
    parser.add_argument("--output_file", type=str, default=None, help="Path for the output .npz file. Defaults to a standard path in the AL cache.")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size for embedding generation.")
    args = parser.parse_args()

    al_cfg = ALConfig()
    model_name = os.path.basename(al_cfg.base_model_path.strip('/'))

    # Determine output paths
    output_dir = os.path.join(al_cfg.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = args.output_file or os.path.join(output_dir, "problem_embeddings.npz")
    error_log_path = output_path.replace('.npz', '_errors.jsonl')

    print("--- Starting Problem Embedding Generation Task ---")
    print(f"Model: {model_name}")
    print(f"Output embeddings will be saved to: {output_path}")

    # Load problem data pool
    if not os.path.exists(al_cfg.full_data_pool_path):
        print(f"Error: Data pool file not found -> {al_cfg.full_data_pool_path}", file=sys.stderr)
        sys.exit(1)

    with open(al_cfg.full_data_pool_path, 'r', encoding='utf-8') as f:
        problems_to_process = list(set([json.loads(line)['problem'] for line in f]))
    print(f"Found {len(problems_to_process)} unique problems to process.")

    # Resume from cache if it exists
    cached_problems, cached_embeddings = [], []
    if os.path.exists(output_path):
        try:
            with np.load(output_path, allow_pickle=True) as data:
                cached_problems = data['problems'].tolist()
                cached_embeddings = list(data['embeddings'])
            print(f"Loaded {len(cached_problems)} cached embeddings from '{output_path}'.")
        except Exception as e:
            print(f"Warning: Could not load cache file '{output_path}'. Starting fresh. Error: {e}")

    processed_problem_set = set(cached_problems)
    problems_to_process = [p for p in problems_to_process if p not in processed_problem_set]

    if not problems_to_process:
        print("All problems already have embeddings. Task complete.")
        return
    print(f"{len(problems_to_process)} new problems need embedding.")

    # Load model
    print(f"Loading model from: {al_cfg.base_model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(al_cfg.base_model_path)
        model = AutoModel.from_pretrained(al_cfg.base_model_path, torch_dtype=torch.bfloat16, device_map="auto")
        model.eval()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load the model. Details: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate embeddings in batches
    problem_batches = [problems_to_process[i:i + args.batch_size] for i in range(0, len(problems_to_process), args.batch_size)]
    failed_batches = 0
    with tqdm(total=len(problems_to_process), desc="Generating Embeddings") as pbar:
        for batch in problem_batches:
            batch_results = get_embeddings_batched(
                problems=batch, model=model, tokenizer=tokenizer
            )

            if batch_results:
                cached_problems.extend(list(batch_results.keys()))
                cached_embeddings.extend(list(batch_results.values()))
                # Save after each successful batch to prevent data loss
                try:
                    np.savez_compressed(
                        output_path,
                        problems=np.array(cached_problems, dtype=object),
                        embeddings=np.array(cached_embeddings)
                    )
                except Exception as e:
                    print(f"Error: Failed to save cache to '{output_path}': {e}", file=sys.stderr)
            else:
                failed_batches += 1
                with open(error_log_path, 'a', encoding='utf-8') as f_err:
                    for p in batch:
                        f_err.write(json.dumps({"failed_prompt_for_embedding": p}) + '\n')

            pbar.update(len(batch))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n--- Embedding Generation Complete ---")
    print(f"Total problems processed: {len(cached_problems)}")
    if failed_batches > 0:
        print(f"Warning: {failed_batches} batches failed. Details of failed prompts are in '{error_log_path}'.")
    print(f"Final embeddings saved to '{output_path}'.")


if __name__ == "__main__":
    main()
