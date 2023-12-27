# Pretraining ViT on ImageNet
The code is modelled after the image classification example from [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-classification)

## Examples
```bash
torchrun --nproc_per_node 4 -m  run_image_classification \
    --dataset_name imagenet \
    --output_dir results/imagenet_outputs/ \
    --remove_unused_columns False \
    --do_train \
    --do_eval \
    --push_to_hub \
    --push_to_hub_model_id vit-base \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --seed 1337 \
    --task pretraining \
    --async_grad
```

## Results
TBD and coming soon