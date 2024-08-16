from strategy import Strategy
from utils.model_agg import fed_average


class FedAvg(Strategy):
    def __init__(self):
        super().__init__()

    def aggregate(self, clients_set, dataset_len_dict, version, output_dir, clients_weights_dict=None):
        return fed_average(clients_set, output_dir, dataset_len_dict, version, clients_weights_dict)

