#!/bin/bash
set -x

CHECKPOINTS_DIR="./checkpoints/" # TODO: change to your own path

export WANDB_MODE=offline

python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=data/train/Qwen2.5-Math-1.5B/CONST_k8.parquet \
 data.val_files=data/test/math500.parquet \
 data.train_batch_size=64 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path='./qwen/Qwen2.5-Math-1.5B' \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.6 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=False \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='al'\
 trainer.experiment_name='Qwen2.5-Math-1.5B-CONST_k8'\
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 +trainer.val_before_train=False \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_epochs=500 2>&1 | tee al.log