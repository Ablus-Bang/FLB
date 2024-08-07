from omegaconf import OmegaConf
import copy
import sys
from os import path
from client.client import BaseClient
from tqdm import tqdm
from server.server import get_model_and_tokenizer, BaseServer

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from peft import get_peft_model_state_dict
from utils.process_data import build_dataset


# load config file, init model and tokenizer
CFG_PATH = "config.yaml"
config_detail = OmegaConf.load(CFG_PATH)
model, tokenizer = get_model_and_tokenizer(CFG_PATH)
model.print_trainable_parameters()
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
if config_detail.sft.training_arguments.gradient_checkpointing:
    model.enable_input_require_grads()

# get dataset
train_dataset_list, test_dataset = build_dataset(
    tokenizer,
    config_detail.dataset_name,
    config_detail.num_clients,
    0.1,
    config_detail.sft.training_arguments.seed,
    config_detail.sft.dataset_sample,
)

# get model weight
global_dict = copy.deepcopy(get_peft_model_state_dict(model))

# init server
local_server = BaseServer(CFG_PATH, model)

training_loss = [[] for i in range(local_server.num_clients)]
for round in tqdm(range(local_server.num_rounds)):
    selected_clients_set = []
    local_dataset_len_dict = dict()
    print(f"start {round+1} rounds")
    for client_id in range(local_server.num_clients):
        # init client
        selected_clients_set.append(client_id)

        local_client = BaseClient(
            client_id,
            CFG_PATH,
            model,
            tokenizer,
            train_dataset_list[client_id],
            test_dataset,
        )

        local_client.local_trainer_set(round + 1)

        local_client.initiate_local_training()

        train_result = local_client.train()

        (model, local_dataset_len_dict) = local_client.save(
            round, local_dataset_len_dict
        )

        training_loss.append(train_result)

        del local_client

    # aggregate model weight
    model = local_server.aggregate(
        selected_clients_set,
        local_dataset_len_dict,
        round,
    )

local_server.save_model()
