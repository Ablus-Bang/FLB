# FLB - Project on the Integration of Federated Learning and Blockchain 

Language: [[English]](README.md) [[中文]](docs/README-cn.md)

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
> Right now we only support CUDA to do training, if you want to use cpu, please set **quantization** to **null** and set **device_map** to **cpu** in `config.yaml`.


## Federate learning finetune

To start the federated learning fine-tuning process, run the following command:

```
python main_simple_fl.py
```

> The client side uses local differential privacy by default, you can close it by setting **local_dp** in client block to **False** in `config.yaml`
> 
> Now in server side, we have only two strategies under `server/strategy/`. The default strategy is federate average. If you want to use differential privacy in server side, you can set **local_dp** in client block to **False** in `config.yaml` and set dp strategy for your server in `main_simple_fl.py`

## More LLM Examples

Provides fine-tuning processes for open source large models to simplify the deployment, use, and application processes of open source large models.

For the web demo built with streamlit, you can run the program directly using the following command


command:
```
streamlit run examples/xxx/xxx-web-demo.py --server.address 127.0.0.1 --server.port 8080
```

## Support Blockchain

We set a contract project to support data record and reward distribution in blockchain.

Right now we use **number of times to upload weights * size of weight file uploaded each time + total training data volume** as the main factor rewards for each client.

> You can set **chain_record** in `config.yaml` to **False** if you don't want to connect with blockchain. The default value of **chain_record** is **False**.

In order to interact with the blockchain, you need to prepare the `node.js v20.0.x` environment

### Create an account
Two roles are required in the interaction with the chain: administrator and user. The administrator is the creator of the smart contract and is responsible for updating the user score to the chain; the user as a client is a participant of the blockchain and can upload model weight to the chain. Generate administrator and user accounts by executing the scripts `createAdmin.js` and `createAccount.js`:
```shell
cd ./chain/contract/sample
# Install Dependencies
npm install
node createAdmin.js

node createAccount.js
```
After execution, the account information of the administrator and user, including the private key and address, will be printed on the console. You need to copy the private key and address to the corresponding location of the `chain/config.json` file.

> When you need to run a real environment with multiple devices (multiple clients), you can generate multiple users by executing the script `createAccount.js` multiple times, and configure the generated user address and private key to the `chain/config.json` of the corresponding **client** end.

### Get test tokens
In the process of interacting with the chain, a certain amount of gas needs to be consumed, so you need to transfer some tokens to the administrator account and the ordinary user account respectively. Get test tokens through this faucet link: https://dev-faucet.cerboai.com/ Just pass in the address and click the button.

### Deploy the contract
Deploy the contract by executing the following command:
```
cd ./chain/contract/sample
node deployContract.js
```
After the execution is completed, the contract address will be printed out in the console. You need to copy the contract address to the corresponding location of the `chain/config.json` file.

## Additional Information

For more details on the project's implementation and usage, please refer to the documentation provided in the repository.

This README serves as a quick start guide to set up and run the integration of Federated Learning and blockchain technology. For any issues or contributions, please refer to the repository's issue tracker and contribution guidelines.

Under Construction .. 
