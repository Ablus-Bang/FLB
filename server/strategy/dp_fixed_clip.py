import os
import torch
from strategy import Strategy
from utils.differential_privacy import add_gaussian_noise, clip_l2_norm


class DpServerFixedClip(Strategy):
    def __init__(self, strategy: Strategy, noise_multiplier: float, clip_threshold: float, device_map: str):
        super().__init__()
        self.strategy = strategy
        self.noise_multiplier = noise_multiplier
        self.clip_threshold = clip_threshold
        self.params_current = None
        self.device_map = device_map

    def aggregate(self, clients_set, dataset_len_dict, version, output_dir, clients_weights_dict=None):
        clients_weights_dict = dict()
        for k, client_id in enumerate(clients_set):
            single_output_dir = os.path.join(
                output_dir,
                str(version),
                "local_output_{}".format(client_id),
                "pytorch_model.bin",
            )
            single_weights = torch.load(single_output_dir)
            single_weights, _ = clip_l2_norm(single_weights,
                                             self.params_current,
                                             self.clip_threshold,
                                             self.device_map)
            clients_weights_dict[client_id] = single_weights

        new_weight = self.strategy.aggregate(clients_set, dataset_len_dict, version, output_dir, clients_weights_dict)
        if new_weight is not None:
            new_weight = add_gaussian_noise(new_weight,
                                            self.noise_multiplier,
                                            self.clip_threshold,
                                            len(clients_set))
        return new_weight
