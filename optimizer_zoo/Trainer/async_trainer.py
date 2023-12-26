from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments
from trl import DPOTrainer, SFTTrainer
from transformers.utils import is_sagemaker_mp_enabled

@dataclass
class AsyncTrainingArguments(TrainingArguments):
    task: bool = field(
        default="pretraining",
        metadata={
            "help": ("generic trainer for finetuning or pretraining with next token prediction")
        }
    )
    async_grad: bool = field(
        default=False,
        metadata={
            "help": ("compute gradient in distributed training and not combine gradient from different workers.")
        },
    )

# wrapping trainer class to make sure gradient is asynced
class AsyncTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator.sync_gradients = None

    def training_step(self, model, inputs):
        # make sure the gradient is not automatically synced
        with model.no_sync():
            model.train()
            inputs = self._prepare_inputs(inputs)

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps

# wrapping SFT trainer class to make sure gradient is asynced
class AsyncSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        # make sure the gradient is not automatically synced
        with model.no_sync():
            model.train()
            inputs = self._prepare_inputs(inputs)

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps

# wrapping DPO trainer class to make sure gradient is asynced
class AsyncDPOTrainer(DPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        #  make sure the gradient is not automatically synced
        with model.no_sync():
            model.train()
            inputs = self._prepare_inputs(inputs)

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps
