import argparse
from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from client.client import BaseClient
from omegaconf import OmegaConf


CFG_PATH = "config.yaml"
config_detail = OmegaConf.load(CFG_PATH)


def run(client_id):
    client = BaseClient(client_id, CFG_PATH)
    client.run_grpc_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run local client"
    )
    parser.add_argument("--client_id", required=False, type=str, default='12344')
    args = parser.parse_args()

    run(args.client_id)


