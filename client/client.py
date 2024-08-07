from transformers import TrainingArguments
from trl import SFTTrainer
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from collections import OrderedDict
from omegaconf import OmegaConf
import math
import copy
import os
import torch


def cosine_lr(
    current_round: int,
    total_round: int,
    learning_rate_max: float = 0.001,
    learning_rate_min: float = 0.0,
) -> float:
    """Implement cosine learning rate."""
    cos_inner = math.pi * current_round / total_round
    return learning_rate_min + 0.5 * (learning_rate_max - learning_rate_min) * (
        1 + math.cos(cos_inner)
    )


class BaseClient:
    def __init__(
        self, client_id, cfg_path, model, tokenizer, train_dataset, test_dataset
    ):
        self.config_detail = OmegaConf.load(cfg_path)
        self.model = model
        self.client_id = client_id
        self.tokenizer = tokenizer
        self.training_args = TrainingArguments(
            **self.config_detail.sft.training_arguments
        )
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict(
                (name, param.detach())
                for name, param in self.model.named_parameters()
                if "default" in name
            )
        )
        self.params_dict_new = OrderedDict(
            (name, param.detach())
            for name, param in self.model.named_parameters()
            if "default" in name
        )
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def local_trainer_set(self, current_round):
        new_lr = cosine_lr(
            int(current_round),
            self.config_detail.num_rounds,
            self.config_detail.sft.learning_rate_max,
            self.config_detail.sft.learning_rate_min,
        )
        self.training_args.learning_rate = new_lr
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            max_seq_length=self.config_detail.sft.max_seq_length,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            dataset_text_field="text",
        )

    def train(self):
        results = self.trainer.train()
        return results.training_loss

    def save(self, current_round, local_dataset_len_dict):
        local_dataset_len_dict[self.client_id] = len(self.train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(
            self.training_args.output_dir,
            str(current_round),
            "local_output_{}".format(self.client_id),
        )
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(
            self.model, self.params_dict_old, "default"
        )
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")

        return self.model, local_dataset_len_dict
