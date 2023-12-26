
import torch
import transformers
from optimizer_zoo.Optimizer.distributed_lion import DistributedLion
def create_optimizer(model, training_args):
    if training_args.optim == "distributed_lion":
        optimizer = DistributedLion(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        optimizers = (optimizer, transformers.get_cosine_schedule_with_warmup(optimizer, training_args.warmup_steps, training_args.max_steps))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay = training_args.weight_decay)
        optimizers = (optimizer, transformers.get_cosine_schedule_with_warmup(optimizer, training_args.warmup_steps, training_args.max_steps))
    return optimizers
