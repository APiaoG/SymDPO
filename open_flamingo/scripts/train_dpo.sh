#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=8 open_flamingo/train/train_dpo.py   \
    --lm_path path_to_anas-awadalla/mpt-1b-redpajama-200b-dolly   \
    --tokenizer_path path_to_anas-awadalla/mpt-1b-redpajama-200b-dolly   \
    --model_path path_to_checkpoint.pt    \
    --data_path path_to_dpo_dataset  \
    --image_folder path_to_image_folder   \
    --cross_attn_every_n_layers 1   \
    --dataset_resampled   \
    --run_name OpenFlamingo-DPO   \
    --report_to_wandb  \
    --train_num_samples 10000  \
    --learning_rate 5e-6  \
    --num_train_epochs 1  \
    --save_strategy steps   \
    --save_steps 1250  \
    --logging_steps 10 \
    --lr_scheduler_type linear \
    --optim adamw_torch  \
    --model_max_length 2048 \
    --bits 16 \
    --dpo_alpha 1.0 \
    --beta 0.1 \
    --gamma 1.0 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_total_limit 1 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05
