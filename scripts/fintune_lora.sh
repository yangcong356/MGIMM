#!/bin/bash
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

# --pretrain_mm_mlp_adapter ./checkpoints/BMGPG-phi2-2.3b-pretrain-b-64-e-10/mm_prompt_projector.bin\
# --model_name_or_path /data1/users/yangcong/.cache/huggingface/hub/models--meta--llama--Llama-2-7b/llama-2-7b-hf \
# --model_name_or_path microsoft/phi-2 lmsys/vicuna-7b-v1.5 \
    # --pretrain_mm_mlp_adapter ./checkpoints/BMGPG-llama2-7b-pretrain-b-64-e-10/mm_prompt_projector.bin\
    # --pretrain_mm_mlp_adapter ./checkpoints/BMGPG-vicuna-7b-pretrain-b-64-e-10/mm_prompt_projector.bin \

deepspeed --include localhost:0,1,2,3 --master_port 29502 train_mem.py \
    --lora_enable True --lora_r 16 --lora_alpha 256 --mm_projector_lr 2e-6 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data1/users/yangcong/.cache/huggingface/hub/models--meta--llama--Llama-2-7b/llama-2-7b-hf \
    --version llama_2 \
    --dataset_config ./configs/diorgpg.yaml \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --pretrain_mm_mlp_adapter ./checkpoints/BMGPG-llama2-7b-pretrain-b-64-e-10/mm_prompt_projector.bin \
    --bf16 True \
    --output_dir ./checkpoints/BMGPG-llama2-7b-lora-r16-b1-e5 \
    --stage "stage2" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
