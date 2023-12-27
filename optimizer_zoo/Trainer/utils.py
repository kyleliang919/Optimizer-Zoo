from transformers import Trainer, Seq2SeqTrainer
from trl import SFTTrainer, DPOTrainer
from .async_trainer import AsyncTrainer, AsyncSFTTrainer, AsyncDPOTrainer, AsyncSeq2SeqTrainer
def create_trainer(training_args):
    if training_args.task == "pretraining":
        return AsyncTrainer if training_args.async_grad else Trainer
    elif training_args.task == "sft":
        return AsyncSFTTrainer if training_args.async_grad else SFTTrainer
    elif training_args.task == "dpo":
        return AsyncDPOTrainer if training_args.async_grad else DPOTrainer
    elif training_args.task == "seq2seq":
        return AsyncSeq2SeqTrainer if training_args.async_grad else Seq2SeqTrainer
    else:
        NotImplementedError
