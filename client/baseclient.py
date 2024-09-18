from abc import ABC, abstractmethod
from os import path
from omegaconf import OmegaConf
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


class BaseClient(ABC):
    def __init__(self, client_id, cfg_path):
        self.cfg_path = cfg_path
        self.config_detail = OmegaConf.load(cfg_path)
        self.client_id = client_id

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def run_grpc_client(self):
        raise NotImplementedError()
