MASTER_ADDR=127.0.0.1
MASTER_PORT=12346
NUM_MACHINES=1
MACHINE_RANK=0

accelerate launch \
    --config_file configs/fsdp_config_llama3.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $MACHINE_RANK \
    --num_processes 8 \
    --num_machines ${NUM_MACHINES} \
    --mixed_precision bf16 \
    train.py \
    --seed 100 \
    --model_name "meta-llama/Meta-Llama-3-70B" \
    --dataset_name "smangrul/code-chat-assistant-v1" \
    --chat_template_format "none" \
    --add_special_tokens False \
    --append_concat_token False \
    --splits "train,test" \
    --max_seq_len 2048 \
    --max_steps 500 \
    --logging_steps 25 \
    --log_level "info" \
    --eval_steps 100 \
    --save_steps 250 \
    --logging_strategy "steps" \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --bf16 True \
    --packing True \
    --learning_rate 5e-5 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1.0 \
    --output_dir "./shared_storage/sourab/experiments/Meta-Llama-3-70B-asst" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --use_reentrant False \
    --dataset_text_field "content" \
    --use_flash_attn True \
    --ddp_timeout 5400 \
    --optim adamw_torch
    # --optim adamw_8bit 
    # --optim paged_adamw_8bit 
    # 
    
    

#     --push_to_hub \
#     --hub_private_repo True \
#     --hub_strategy "every_save" \
#     --model_name "meta-llama/Llama-2-70b-chat-hf" \
