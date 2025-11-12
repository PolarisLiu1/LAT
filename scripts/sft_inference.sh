export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_NAME="wiki"
RUN_NAME="Qwen2.5-VL-7B_inf_${DATA_NAME}"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12389" \
    ../src/sft_inference.py \
    --deepspeed ../zero3.json \
    --output_dir ../output/$RUN_NAME \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --lora_name_or_path path to model_stageI \
    --dataset_name visa \
    --learning_rate 1e-4 \
    --lr_scheduler_type="cosine" \
    --num_train_epochs 1 \
    --data_seed 3407 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --logging_steps 1 \
    --report_to none \
    --dataset_name $DATA_NAME \
    --single_image true \
    --all_data false \
    --save_steps 500 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2