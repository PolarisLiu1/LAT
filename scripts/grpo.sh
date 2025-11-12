export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_NAME="wiki"

RUN_NAME="Qwen2.5-VL-7B-GRPO-$DATA_NAME"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ../src/grpo.py \
    --deepspeed ../zero3.json \
    --output_dir ../output/$RUN_NAME \
    --model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct"  \
    --dataset_name visa \
    --max_prompt_length 16384 \
    --max_completion_length 600 \
    --num_generations 6 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 3407 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --learning_rate 5e-5 \
    --lr_scheduler_type="cosine" \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --dataset_name $DATA_NAME \
    --single_image true \
    --all_data false \
    --save_steps 50 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_task_type CAUSAL_LM \
    --lora_dropout 0.05 \
    --num_iterations 1
