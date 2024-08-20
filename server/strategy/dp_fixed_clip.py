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

    def aggregate(self, client_list, dataset_len_list, weight_path_list, clients_weights_dict=None):
        clients_weights_dict = dict()
        for k, p in enumerate(weight_path_list):
            single_output_dir = os.path.join(
                p,
                "pytorch_model.bin",
            )
            single_weights = torch.load(single_output_dir)
            single_weights, _ = clip_l2_norm(single_weights,
                                             self.params_current,
                                             self.clip_threshold,
                                             self.device_map)
            clients_weights_dict[p] = single_weights
        # for k, client_id in enumerate(client_list):
        #     single_output_dir = os.path.join(
        #         output_dir,
        #         str(version),
        #         "local_output_{}".format(client_id),
        #         "pytorch_model.bin",
        #     )
        #     single_weights = torch.load(single_output_dir)
        #     single_weights, _ = clip_l2_norm(single_weights,
        #                                      self.params_current,
        #                                      self.clip_threshold,
        #                                      self.device_map)
        #     clients_weights_dict[client_id] = single_weights

        new_weight = self.strategy.aggregate(dataset_len_list, weight_path_list, clients_weights_dict)
        if new_weight is not None:
            new_weight = add_gaussian_noise(new_weight,
                                            self.noise_multiplier,
                                            self.clip_threshold,
                                            len(client_list))
        return new_weight
