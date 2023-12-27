# Finetuning Llama with DPO
The code is modeled after the [huggingface blog post](https://huggingface.co/blog/dpo-trl).
## Examples
Run the SFT experiment on llama7B and stack-exchanged-pair and prepare checkpoint for next step
    
    ```bash
    torchrun --nproc_per_node 4 -m benchmarks/llama2_dpo/sft_llama2 \
    --output_dir="./sft" \
    --max_steps=500 \
    --logging_steps=10 \
    --save_steps=10 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --group_by_length=False \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --bf16=True \
    --remove_unused_columns=False \
    --run_name="sft_llama2" \
    --report_to="wandb" \
    --optim distributed_lion \
    --task sft \
    --async_grad
    ```

Run DPO

    ```bash
    torchrun --nproc_per_node 4 -m benchmarks/llama2_dpo/dpo_llama2 \
         --model_name_or_path="sft/final_checkpoint" \
         --output_dir="dpo" \
         --optim distributed_lion \
         --task dpo \
         --async_grad
    ```

## Result
TBD coming soon