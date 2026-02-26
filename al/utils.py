# al/utils.py
import torch
import re
import math
from omegaconf import OmegaConf
from typing import List, Dict
from vllm import LLM, SamplingParams
import ray


def extract_boxed_answer(solution_text: str) -> str:
    """Extracts the final answer from a LaTeX \boxed{} environment in a full solution text."""
    if not isinstance(solution_text, str):
        return ""
    starts = list(re.finditer(r"\\boxed\{", solution_text))
    if not starts:
        return ""

    last_start_match = starts[-1]
    content_start_index = last_start_match.end()

    brace_level = 1
    answer_content = []
    for char in solution_text[content_start_index:]:
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1

        if brace_level == 0:
            break
        answer_content.append(char)

    return "".join(answer_content).strip() if brace_level == 0 else ""


@ray.remote
class VLLMSamplingActor:
    """A Ray Actor that encapsulates a vLLM engine for distributed, high-performance inference."""
    def __init__(self, model_path: str, rollout_config: Dict, instruction_prompt: str, answer_extraction_prompt: str):
        self.model_path = model_path
        self.rollout_config = rollout_config
        self.llm_engine = None
        self.instruction_prompt = instruction_prompt
        self.answer_extraction_prompt = answer_extraction_prompt

    def initialize_vllm_engine(self):
        """Initializes the vLLM engine."""
        self.config = self.rollout_config
        tp_size = self.config.get("tensor_model_parallel_size", 1)
        max_len = self.config.get("max_model_len", 4096)

        self.llm_engine = LLM(
            model=self.model_path,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=self.config.get("gpu_memory_utilization", 0.9),
            dtype=self.config.get("dtype", "bfloat16"),
            max_model_len=max_len,
            distributed_executor_backend=self.config.get("distributed_executor_backend"),
            enforce_eager=self.config.get("enforce_eager", False),
            seed=42
        )
        return True

    @torch.no_grad()
    def get_outcome_volatility_answers(self, prompts: List[str]) -> Dict[str, List[str]]:
        """Generates n independent answers for each prompt (Outcome Volatility)."""
        if self.llm_engine is None:
            raise RuntimeError("VLLM engine not initialized.")
        if not prompts:
            return {}

        sampling_params = SamplingParams(
            n=self.rollout_config.get('n', 20),  # Use n from config
            temperature=self.rollout_config.get('temperature', 0.6),
            max_tokens=self.rollout_config.get('response_length', 2048)
        )
        full_prompts = [p + self.instruction_prompt for p in prompts]
        outputs = self.llm_engine.generate(full_prompts, sampling_params, use_tqdm=False)

        results = {p: [] for p in prompts}
        for output in outputs:
            original_prompt = output.prompt.replace(self.instruction_prompt, "")
            answers = [extract_boxed_answer(sample.text) for sample in output.outputs]
            results[original_prompt] = answers
        return results

    def get_procedural_volatility_answers(self, prompts: List[str], num_checkpoints: int) -> Dict[str, List[str]]:
        """Extracts answers from various checkpoints along a single reasoning trajectory (Procedural Volatility)."""
        if self.llm_engine is None:
            raise RuntimeError("VLLM engine not initialized.")
        if not prompts:
            return {}

        # Stage 1: Generate long reasoning trajectories
        trajectory_params = SamplingParams(n=1, temperature=0, max_tokens=self.rollout_config.get('response_length', 2048))
        full_prompts = [p + self.instruction_prompt for p in prompts]
        trajectory_outputs = self.llm_engine.generate(full_prompts, trajectory_params, use_tqdm=False)

        # Stage 2: Prepare prompts for answer extraction from truncated trajectories
        extraction_prompts, prompt_metadata = [], []
        for output in trajectory_outputs:
            original_prompt = output.prompt.replace(self.instruction_prompt, "")
            full_trajectory_tokens = output.outputs[0].token_ids
            total_tokens = len(full_trajectory_tokens)

            for i in range(1, num_checkpoints + 1):
                token_checkpoint = math.ceil(total_tokens * (i / num_checkpoints))
                if token_checkpoint > 0:
                    partial_trajectory_text = self.llm_engine.get_tokenizer().decode(full_trajectory_tokens[:token_checkpoint])

                    extraction_prompt = (
                        f"{self.answer_extraction_prompt}"
                        f"Reasoning:\n{original_prompt}{self.instruction_prompt}{partial_trajectory_text}\n"
                        f"Your question:\nBased on the reasoning above, what is the final answer?\n\n"
                        f"Your response:\n"
                    )
                    extraction_prompts.append(extraction_prompt)
                    prompt_metadata.append(original_prompt)

        if not extraction_prompts:
            return {p: [] for p in prompts}

        # Stage 3: Batch predict answers from partial trajectories
        extraction_params = SamplingParams(n=1, temperature=0, max_tokens=512)
        extraction_outputs = self.llm_engine.generate(extraction_prompts, extraction_params, use_tqdm=False)

        # Stage 4: Collect and organize the final answers
        final_results = {p: [] for p in prompts}
        for i, output in enumerate(extraction_outputs):
            original_prompt = prompt_metadata[i]
            predicted_answer = extract_boxed_answer(output.outputs[0].text)
            final_results[original_prompt].append(predicted_answer)

        return final_results
