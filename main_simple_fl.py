from omegaconf import OmegaConf
from threading import Thread
import sys
from os import path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from client.client import BaseClient
from server.server import BaseServer


# load config file, init model and tokenizer
CFG_PATH = "config.yaml"
config_detail = OmegaConf.load(CFG_PATH)


def runserver():
    server = BaseServer(CFG_PATH)
    server.start()


def run_simulation():
    server_thread = Thread(target=runserver)
    server_thread.start()

    for i in range(config_detail.num_clients):
        client = BaseClient(i, CFG_PATH)
        client.start()
        del client

    server_thread.join()


if __name__ == "__main__":
    run_simulation()
