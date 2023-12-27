# Continuous pretraining of GPT-2
The code is modeled after huggingface [exapmle](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)

## Command
Run the pretraining experiment on gpt2 and openwebtext:

    ```bash
    torchrun --nproc_per_node 4 -m benchmarks/gpt2_pretraining/run_clm \
    --config_name gpt2 \
    --tokenizer_name gpt2 \
    --dataset_name openwebtext \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --output_dir result/gpt2_lion_wd_0.1 \
    --report_to wandb \
    --torch_dtype bfloat16 \
    --gradient_accumulation_steps 8 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --optim distributed_lion \
    --save_total_limit 2 \
    --learning_rate 0.0001 \
    --weight_decay 0.1 \
    --task pretraining \
    --async_grad
    ```

## Results
TBD, coming soon