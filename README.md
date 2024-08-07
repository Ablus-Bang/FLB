# FLB - Project on the Integration of Federated Learning and Blockchain 

## Overview

This project aims to integrate Federated Learning (FL) with blockchain technology, enhancing data privacy and security in machine learning applications. Federated Learning enables training a global model across decentralized devices holding local data samples without exchanging them. By incorporating blockchain, we can ensure a tamper-proof, transparent, and secure environment for federated learning processes.


## Setup
Clone the repo, prepare local environment：
```
conda create -n flb python=3.11
conda activate flb
pip install -r requirements.txt
```

## Modify config file
In our initial **config.yaml** file, you need to set your local model path and dataset_name. <br>
> Right now we only support CUDA to do training, if you want to use cpu, please delete **quantization** in config.yaml and set **device_map** to cpu.


## Federate learning finetune

To start the federated learning fine-tuning process, run the following command:

```
python main_simple_fl.py
```

## Support Blockchain

We dsign a basic blockchain protocol (`utils/blockchain.py`) to measure and reward clients in our federated learning setup. We'll use the number of parameters as the main factor for rewards.

Under Construction .. 

## Additional Information

For more details on the project's implementation and usage, please refer to the documentation provided in the repository.

This README serves as a quick start guide to set up and run the integration of Federated Learning and blockchain technology. For any issues or contributions, please refer to the repository's issue tracker and contribution guidelines.

Under Construction .. 
