from abc import ABC, abstractmethod


class Strategy(ABC):

    @abstractmethod
    def aggregate(self, clients_set, dataset_len_dict, version, output_dir, clients_weights_dict=None):
        """Aggregate results from clients

        :param clients_set: Clients selected list
        :param dataset_len_dict: Each client dataset length
        :param version: current server model version
        :param output_dir: file output dir
        :param clients_weights_dict: clients weights dict
        :return: new model parameters
        """
        pass
