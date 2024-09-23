#!/bin/bash 

CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 icvrc/run_pt_train.py \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-mt0-v3 \
    --output_dir /code/models/vivrc_vlsp_mblip_mt0_xl_model \
    --overwrite_output_dir \
    --do_train \
    --data_folder /code/data \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 5e-5 \
    --bf16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --logging_steps 10 \
    --num_train_epochs 4 \
    --deepspeed deepspeed_config.json
