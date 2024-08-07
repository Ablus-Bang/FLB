from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize


def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    r"""
    selected_clients_set:选中客户端的集合列表
    output_dir: 权重输出文件夹
    local_dataset_len_dict: 每个客户端的数据集大小
    epoch:训练轮次
    """
    weights_array = normalize(
        torch.tensor(
            [local_dataset_len_dict[client_id] for client_id in selected_clients_set],
            dtype=torch.float32,
        ),
        p=1,
        dim=0,
    )
    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(
            output_dir,
            str(epoch),
            "local_output_{}".format(client_id),
            "pytorch_model.bin",
        )
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {
                key: single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }
        else:
            weighted_single_weights = {
                key: weighted_single_weights[key]
                + single_weights[key] * (weights_array[k])
                for key in single_weights.keys()
            }

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model
