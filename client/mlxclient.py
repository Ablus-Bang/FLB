import os
from os import path
import math
import sys
import json

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from datetime import datetime

from pathlib import Path
import types
from .baseclient import BaseClient
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pickle
import socket
from utils.chain_record import send_weight
from utils.tokenizer_utils import TokenizerWrapper
from utils.tuner.datasets import load_dataset
from utils.tuner.trainer import TrainingArgs, TrainingCallback, evaluate, train
from utils.tuner.utils import (
    apply_lora_layers,
    build_schedule,
    linear_to_lora_layers,
    print_trainable_parameters,
)
from utils.mlx_utils import load, save_config

def train_model(
        args,
        model: nn.Module,
        tokenizer: TokenizerWrapper,
        train_set,
        valid_set,
        client_id,
        training_callback: TrainingCallback = None,
    ):
        # Freeze all layers
        model.freeze()

        # Convert linear layers to lora layers and unfreeze in the process
        linear_to_lora_layers(model, args.lora_layers, args.lora_parameters, args.use_dora)

        # Resume training the given adapters.
        if args.resume_adapter_file is not None:
            print(f"Loading pretrained adapters from {args.resume_adapter_file}")
            model.load_weights(args.resume_adapter_file, strict=False)

        print_trainable_parameters(model)

        adapter_path = Path(args.adapter_path+f"/local_client_{client_id}")
        adapter_path.mkdir(parents=True, exist_ok=True)
        adapter_file = adapter_path / "adapters.safetensors"
        save_config(vars(args), adapter_path / "adapter_config.json")

        # init training args
        training_args = TrainingArgs(
            batch_size=args.batch_size,
            iters=args.iters,
            val_batches=args.val_batches,
            steps_per_report=args.steps_per_report,
            steps_per_eval=args.steps_per_eval,
            steps_per_save=args.save_every,
            adapter_file=adapter_file,
            max_seq_length=args.max_seq_length,
            grad_checkpoint=args.grad_checkpoint,
        )

        model.train()
        opt = optim.Adam(
            learning_rate=(
                build_schedule(args.lr_schedule) if args.lr_schedule else args.learning_rate
            )
        )
        # Train model
        weights = train(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            optimizer=opt,
            train_dataset=train_set,
            val_dataset=valid_set,
            training_callback=training_callback,
        )
        return weights,vars(args)


def evaluate_model(args, model: nn.Module, tokenizer: TokenizerWrapper, test_set):
    model.eval()

    test_loss = evaluate(
        model=model,
        dataset=test_set,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_batches=args.test_batches,
        max_seq_length=args.max_seq_length,
    )

    test_ppl = math.exp(test_loss)

    print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")


class MLXClient(BaseClient):

    def __init__(self, client_id, cfg_path):
        super().__init__(client_id, cfg_path)
        config={
                "model": self.config_detail.model.model_path,
                "train": self.config_detail.mlx.train,
                "data": self.config_detail.dataset_name,
                "seed": self.config_detail.mlx.seed,
                "lora_layers": self.config_detail.mlx.lora_layers,
                "batch_size": self.config_detail.mlx.train_arg.batch_size,
                "iters": self.config_detail.mlx.train_arg.iters,
                "val_batches": self.config_detail.mlx.train_arg.val_batches,
                "learning_rate": self.config_detail.mlx.learning_rate,
                "steps_per_report": self.config_detail.mlx.train_arg.steps_per_report,
                "steps_per_eval": self.config_detail.mlx.train_arg.steps_per_eval,
                "resume_adapter_file": self.config_detail.mlx.resume_adapter_file,
                "adapter_path": self.config_detail.mlx.adapter_path,
                "save_every": self.config_detail.mlx.train_arg.save_every,
                "test": self.config_detail.mlx.test,
                "test_batches": self.config_detail.mlx.test_batches,
                "max_seq_length": self.config_detail.mlx.train_arg.max_seq_length,
                "lr_schedule": self.config_detail.mlx.lr_schedule,
                "lora_parameters": {
                    "rank": self.config_detail.model.lora.peft_lora_r,
                    "alpha": self.config_detail.model.lora.peft_lora_alpha,
                    "dropout": 0.0,
                    "scale": 20.0
                },
                "use_dora": self.config_detail.mlx.use_dora,
                "grad_checkpoint": self.config_detail.mlx.train_arg.grad_checkpoint
            }
        self.args = types.SimpleNamespace(**config)
        self.host = self.config_detail.client.host
        self.port = self.config_detail.client.port
        self.use_chain = self.config_detail.chain_record
    
    
    def train(self,training_callback: TrainingCallback = None):
       print(self.args)
       np.random.seed(self.args.seed)
       print("Loading pretrained model")
       model, tokenizer = load(self.args.model)
       print("Loading datasets")
       train_set, valid_set, _ = load_dataset(self.args, tokenizer)
       print("Training")
       weights,config = train_model(self.args, model, tokenizer, train_set, valid_set,self.client_id, training_callback)
       return weights,config,len(train_set)

    def test(self):
        np.random.seed(self.args.seed)
        print("Loading pretrained model")
        model, tokenizer = load(self.args.model)
        print("Loading datasets")
        _, _, test_set = load_dataset(self.args, tokenizer)
        if self.args.adapter_path != "":
            apply_lora_layers(model, self.args.adapter_path)
        evaluate_model(self.args, model, tokenizer, test_set)
        
    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            print(f"Connected to {self.host}:{self.port}")
            new_model_weight,lora_config,train_dataset_len = self.train()
            data = pickle.dumps(
                {
                    "client_id": self.client_id,
                    "train_dataset_length": train_dataset_len,
                    "new_model_weight": new_model_weight,
                    "lora_config": lora_config
                }
            )
            s.sendall(data)
            print("Training complete, weights sent to server")
            if self.use_chain is True:
            # record weight to chain, now just record file path
                current_date = datetime.today().strftime("%Y%m%d_%H%M%S")
                weight_path = os.path.join(
                    self.config_detail.server.clients_file_save_path,
                    "local_output_{}".format(str(self.client_id)),
                    current_date,
                )
                send_weight(weight_path)

    def run_grpc_client(self):
        raise NotImplementedError


if __name__ == "__main__":
    client = MLXClient(client_id="1233", cfg_path="../config.yaml")
    client.start()