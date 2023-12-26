from transformers import Trainer
from trl import SFTTrainer, DPOTrainer
from .async_trainer import AsyncTrainer, AsyncSFTTrainer, AsyncDPOTrainer
def create_trainer(training_args):
    if training_args.task == "pretraining":
        return AsyncTrainer if training_args.async_grad else Trainer
    elif training_args.task == "sft":
        return AsyncSFTTrainer if training_args.async_grad else SFTTrainer
    elif training_args.task == "dpo":
        return AsyncDPOTrainer if training_args.async_grad else DPOTrainer
    else:
        NotImplementedError
