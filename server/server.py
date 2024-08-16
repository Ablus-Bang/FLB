from omegaconf import OmegaConf
import torch
import os
from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from strategy.strategy import Strategy
from strategy.fedavg import FedAvg
import socket
import pickle
import redis
import time
import uuid


class BaseServer:
    def __init__(self, cfg_path: str, strategy: Strategy = None):
        self.config_detail = OmegaConf.load(cfg_path)
        self.cfg_path = cfg_path
        self.model_parameter = None
        self.num_clients = self.config_detail.num_clients
        self.num_rounds = self.config_detail.num_rounds
        self.host = self.config_detail.server.host
        self.port = self.config_detail.server.port
        self.redis_client = redis.from_url(self.config_detail.server.redis_url)
        self.save_path = "./save"
        self.strategy = strategy if strategy is not None else FedAvg()

    def aggregate(self, clients_set, dataset_len_dict):
        if clients_set > 0:
            curr_version = self.redis_client.get(
                f"{self.config_detail.model.model_path}_version"
            )
            self.model_parameter = self.strategy.aggregate(clients_set,
                                                           dataset_len_dict,
                                                           curr_version,
                                                           self.config_detail.sft.training_arguments.output_dir)

    def save_model(self):
        new_version = int(time.time())
        torch.save(
            self.model_parameter,
            os.path.join(
                self.config_detail.sft.training_arguments.output_dir,
                str(new_version),
                "adapter_model.bin",
            ),
        )
        self.redis_client.set(
            f"{self.config_detail.model.model_path}_version", new_version
        )

    def start(self):
        current_save_version = str(uuid.uuid1())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"Server listening on {self.host}:{self.port}")

            client_weights = {}
            client_list = []
            for _ in range(self.num_clients):
                conn, addr = s.accept()
                with conn:
                    print(f"Connected by {addr}")
                    # Send current model weights
                    # data = pickle.dumps(self.model.state_dict())
                    # conn.sendall(data)

                    # Receive updated weights
                    data = b""
                    while True:
                        packet = conn.recv(4096)
                        if not packet:
                            break
                        data += packet
                    recv_data = pickle.loads(data)
                    print(
                        f'Received data from client: {recv_data["client_id"]}, data: {recv_data}'
                    )
                    client_list.append(recv_data["client_id"])
                    client_weights[recv_data["client_id"]] = recv_data[
                        "new_model_weight"
                    ]
                    #
                    client_save_path = os.path.join(
                        self.save_path,
                        "client:" + str(recv_data["client_id"]),
                        current_save_version,
                    )
                    os.makedirs(client_save_path, exist_ok=True)
                    torch.save(
                        recv_data["new_model_weight"],
                        client_save_path + "/pytorch_model.bin",
                    )

            # Average the weights
            # self.aggregate(client_list, client_weights)
            # print("Federated learning complete")
            # self.save_model()


if __name__ == "__main__":
    server = BaseServer(cfg_path="../config.yaml")
    server.start()
