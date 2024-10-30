#!/bin/bash 


## 1 - Model mblip-mt0-qa ##
## Model mBLIP MT0 với Question Aware
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-qa \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_qa_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_qa_bs512_lr2e5_10e


## 2 - Model mblip-mt0 ##
## Model mBLIP MT0 nguyên bản
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_bs512_lr2e5_10e


## 3 - Model mblip-mt0-st ##
## Model mBLIP MT0 với Scene text từ ESTextSpotter
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-st \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_st_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_st_bs512_lr2e5_10e


## 4 - Model mblip-mt0-st-qa ##
## Model mBLIP MT0 với Scene text (ESTextSpotter) kết hợp với Question Aware
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-st-qa \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_stqa_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_stqa_bs512_lr2e5_10e



## 5 - Model mblip-mt0-qa-qe ##
## Model mBLIP MT0 với Question Aware, Qformer Embedding được khởi tạo mới
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-qa-qe \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_qaqe_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_qaqe_bs512_lr2e5_10e


## 6 - Model mblip-mt0-st-qa-qe ##
## Model mBLIP MT0 với Scene text (ESTextSpotter) kết hợp với Question Aware (New Embedding for Qformer)
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-st-qa-qe \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_stqaqe_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/estp_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/estp_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_stqaqe_bs512_lr2e5_10e


## 7 - Model mblip-mt0-st ##
## Model mBLIP MT0 với Scene text từ Vintern-1B
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-st \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_st_vintern_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/vintern_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/vintern_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_st_vintern_bs512_lr2e5_10e


## 8 - Model mblip-mt0-st-qa ##
## Model mBLIP MT0 với Scene text (Vintern-1B) kết hợp với Question Aware
deepspeed --include localhost:0,1  --master_port=19998 icvrc/run_pt_train.py \
    --deepspeed deepspeed_config.json \
    --model_name_or_path Gregor/mblip-mt0-xl \
    --model_type mblip-vqa-st-qa \
    --output_dir mounts/models/vivrc_mblip_mt0xl_model_stqa_vintern_bs512_lr2e5_10e \
    --overwrite_output_dir \
    --do_train \
    --train_image_folder mounts/data/training-images \
    --train_data_file mounts/data/vintern_preprocessed_vlsp2023_train.json \
    --dev_image_folder mounts/data/dev-images \
    --dev_data_file mounts/data/vintern_preprocessed_vlsp2023_dev.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --question_max_len 48 \
    --answer_max_len 64 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --report_to wandb \
    --run_name vivrc_mblip_mt0xl_model_stqa_vintern_bs512_lr2e5_10e


