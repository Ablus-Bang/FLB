# FLB - Project on the Integration of Federated Learning and Blockchain 

Language: [[English]](README.md) [[中文]](docs/README-cn.md)

## Overview

This project aims to integrate Federated Learning (FL) with blockchain technology to enhance data privacy and security in machine learning applications. Federated Learning allows training a global model across decentralized devices that hold local data samples without exchanging them. By combining blockchain, we can ensure that the federated learning process is immutable, transparent, and secure.


## Setup

Clone the project and prepare the local environment:
```
conda create -n flb python=3.11
conda activate flb
pip install -r requirements.txt
```

**Note** If the machine is not part of the Apple M-series chips, installing the mlx and mlx-lm dependency may result in an error. You can choose to comment it out and reinstall.


### CPU
You can use CPU to do training, but we suggest to use GPU for faster training.

### GPU support
If you want to use GPU for training, we recommend NVIDIA RTX 30 series and above graphics cards with at least 8GB of video memory. Memory at least 16GB of RAM.

#### Check the CUDA environment.

Open the command prompt window (such as PowerShell or Command Prompt), and then enter the following command:
```
nvcc --version
```
If CUDA is installed, it will display the version information of the CUDA compiler tools, similar to the following output:

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:19:38_PST_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```

#### Install CUDA
If there is a GPU but no CUDA information, please follow the tutorial below to install CUDA:

[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

[CUDA Installation Guide for Microsoft Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)


### For Mac OS
For Mac OS training, only support Macbook with newer than M2 chips now. 

## Configuring Files
In our initial **config.yaml** file, you need to set your **model path** and **dataset name**. 

```
model:
  model_path:  # model path in Hugging face or local model path
  quantization: 8 # if you want to use cpu, please set this to null
  device_map: "cuda" # support cuda, cpu, mps and mlx for os system
 
dataset_name： # dataset in Hugging face or local dataset path
```


### For OS training config setting
> You must set **device_map** to **mlx**. <br>
> You need to use specific dataset format for mlx training, please refer to **examples/datasets/mlx_datasets** <br>
> You can only use models from [**mlx-community**](https://huggingface.co/mlx-community) in huggingface. Or you can conver your model to specific model structure by using function **convert** in **utils/mlx_utils.py**.
> 
> More parameters setting for mlx training is in **config.yaml** **mlx** part <br>
> ```
> mlx:
>   lora_layers: 16 # Number of layers to fine-tune by using mlx
>   learning_rate: 1e-5 # Adam learning rate.
>   resume_adapter_file: null # Load path to resume training with the given adapter weights.
>   adapter_path: "adapters" # Save/load path for the trained adapter weights.
>   train: True
>   test: false # Evaluate on the test set after training
>   test_batches: 100 # Number of test set batches, -1 uses the entire test set.
>   use_dora: false # Use DoRA instead of LoRA.
>   lr_schedule: null
>   seed: 1212
>   train_arg:
>     batch_size: 4 # Minibatch size.
>     iters: 100 # Iterations to train for.
>     val_batches: 25 # Number of validation batches, -1 uses the entire validation set.
>     steps_per_report: 10 # Number of training steps between loss reporting.
>     steps_per_eval: 50 # Number of training steps between validations.
>     save_every: 50 # Save the model every N iterations.
>     max_seq_length: 2048 # Maximum sequence length.
>     grad_checkpoint: false # Use gradient checkpointing to reduce memory use.
> ```


## Local Federated Learning Fine-tuning

**Efficiency**: We consider the use of Parameter-Efficient Fine-Tuning for local clients, such as LoRA. 

To start the framework test on a machine and simulate the federated learning fine-tuning process, run the following command:

```
python main_fl_socket.py
```

The client side uses [local differential privacy](https://en.wikipedia.org/wiki/Local_differential_privacy) by default, you can close it by setting **local_dp** in client block to **False** in `config.yaml`

If you want to use differential privacy in server side, you can set **local_dp** in client block to **False** in `config.yaml` and run:
 ```
 python main_fl_socket.py --use_server_dp=true
 ```

We also support [gRPC](https://grpc.io/) for client and server communication, you can run this script to simulate:
> For mlx training, gRPC is not supported now, we will support this in the future.
```
python main_fl_grpc.py
```

If you want to use differential privacy in server side, you can set **local_dp** in client block to **False** in `config.yaml` and run:
 ```
 python main_fl_grpc.py --use_server_dp=true
 ```

> We support create an insecure and secure gRPC channel, you can set your local [root certificates](https://en.wikipedia.org/wiki/Root_certificate) to config.yaml to use secure channel:
>  ```
> client:
>   grpc_insecure: True # you can set it to False to turn off it and set grpc_auth_cer_path to use secure gRPC channel
>   grpc_auth_cer_path: null # set your local root certificates path to here
> ```

You can modify [proto file](https://protobuf.dev/getting-started/pythontutorial/) `utils/protos/communicate.proto` to generate your new message structure and communication function.

Now in server side, we have only two strategies under `server/strategy/`, we will add more in the future. 
- [x] Federate average (default strategy in server side)
> The central server aggregates the models received from the clients by averaging the model parameters. 
- [x] Federate average + differential privacy with fixed clipping
> When implementing differential privacy, data often needs to be processed to reduce sensitivity. Fixed clipping is one such method. It refers to clipping or limiting the data according to a preset threshold before it is used for further analysis. The purpose of this is to control the range of the data, thereby reducing the impact of individual extreme values ​​on the final result and ensuring that the output of the algorithm does not pose a risk to the privacy of any individual.
- [ ] Federate average + differential privacy with adaptive clipping
> Different from fixed clipping, adaptive clipping does not pre-set a fixed clipping threshold, but dynamically adjusts the threshold according to the actual distribution of data and the required privacy protection level.
- [ ] ...

### Evaluation
You can use `utils/eval_from_local.py` as a script to evaluate the model.
```
cd utils
python eval_from_local.py
```
You can customize the script by setting parameters, running script with -h to see each parameter description:
```commandline
python eval_from_local.py -h

usage: eval_from_local.py [-h] [--ntrain NTRAIN] [--selected_subjects SELECTED_SUBJECTS] [--save_dir SAVE_DIR] [--lora_config_path LORA_CONFIG_PATH]
                          [--lora_weights_path LORA_WEIGHTS_PATH] [--global_record_file GLOBAL_RECORD_FILE] [--model MODEL]

options:
  -h, --help            show this help message and exit
  --ntrain NTRAIN, -k NTRAIN
                        few-shot examples amount, default is 3
  --selected_subjects SELECTED_SUBJECTS, -sub SELECTED_SUBJECTS
                        selected subjects: biology, business, chemistry, computer science, economics, engineering, health, history, law, math, philosophy, physics, psychology, other,
                        all. default is 'all'
  --save_dir SAVE_DIR, -s SAVE_DIR
                        evaluation results save dir, default is 'eval_results'
  --lora_config_path LORA_CONFIG_PATH, -lc LORA_CONFIG_PATH
                        lora config folder path
  --lora_weights_path LORA_WEIGHTS_PATH, -lw LORA_WEIGHTS_PATH
                        lora weights bin file path
  --global_record_file GLOBAL_RECORD_FILE, -grf GLOBAL_RECORD_FILE
                        global log record file, default is 'eval_record_collection.csv'
  --model MODEL, -m MODEL
                        local model path
```
We use [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) as our dataset to do evaluation.

Also we use evaluation in our server side `server/server.py`：
```python
def eval(self, lora_config_path='', model_weights_path='', clients_data_detail=None):
    results = eval_model(self.config_detail.model.model_path, lora_config_path, model_weights_path, n_train=1)
    result_save_path = os.path.join(self.output, self.latest_version)
    with open(result_save_path + '/eval_result.json', 'w') as f:
        json.dump(results, f)
    print(f'Eval results: {results}')
    # calculate and send reward to each client address
    if self.use_chain is True:
        self.calculate_reward(clients_data_detail)

def update(self, do_eval=True):
    """Aggregate model and save new model weight, send reward to each client"""
    clients_detail, dataset_length_list, path_list = get_clients_uploads_after(self.save_path, self.latest_version)
    client_list = clients_detail.keys()
    if len(client_list) > 0:
        self.aggregate(clients_detail.keys(), dataset_length_list, path_list)
        self.save_model()
        if do_eval:
            weight_saved_path = os.path.join(self.output, self.latest_version, 'adapter_model.bin')
            eval_thread = Thread(target=self.eval,
                                 args=['./output',
                                       weight_saved_path,
                                       clients_detail])
            eval_thread.start()
    else:
        logging.info('No new weights from client')
```

####  Test local server evaluation and save new model weight file <a id="server_side_weight_save"></a>
You can run `run_fast_api.py` and use interface `POST /update_wight` to do evaluation for the model. It will automatically aggregate and save a new model weight file to your local folder `server_output`: 
```commandline
python run_fast_api.py

curl -X 'POST' \
  'http://127.0.0.1:8080/update_wight' \
  -H 'accept: application/json' \
  -d ''
```

## Model Download And Model Update

### Model Download

Download the model from huggingface to a local specified location.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="microsoft/Phi-3-mini-4k-instruct",local_dir="./model", ignore_patterns=["*.gguf"])
```

### Client SFT Finetuning

We use the LoRA method for finetuning models. 

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig,TaskType
from peft.utils import prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import torch
from datasets import Dataset
import pandas as pd

model_path = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="right")
tokenizer.pad_token = tokenizer.unk_token
tokenizer.model_max_length = 2048
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

def apply_chat_template(
    example,
    tokenizer,
):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example
processed_train_dataset = ds.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=ds.column_names,
        desc="Applying chat template to train_sft",
    )

model = model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, 
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.055,
    bias="none",
)
model = get_peft_model(model, config)
args=SFTConfig(
    output_dir="./output/Phi-3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=20,
    log_level="info",
    num_train_epochs=50,
    save_steps=100,
    learning_rate=1e-4,
    save_total_limit=2,
    gradient_checkpointing=True,
    dataset_text_field="text",
    max_seq_length=2048,
)
trainer = SFTTrainer(
            model=model,
            train_dataset=processed_train_dataset,
            tokenizer=tokenizer,
            args=args
        )
trainer.train()
lora_path='./Phi-3_lora'
trainer.model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
```

#### TODO
[]Q-LoRA  
[]Q-Bottleneck Adapters  
[]Q-PrefixTuning  


### Service Model Update
```python
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
import torch
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "./model_path",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
)

tokenizer = AutoTokenizer.from_pretrained("./model_path")
lora_weights_path = "./model_path/output/0/adapter_model.bin"
lora_config_path = "./model_path/output"
config = LoraConfig.from_pretrained(lora_config_path)
lora_weights = torch.load(lora_weights_path)

# load lora adapter
model = PeftModel(model, config)
set_peft_model_state_dict(model,lora_weights)

# merge weight.
model = model.merge_and_unload()
```

#### TODO
[]Weight Upload


## More LLM Examples

[√] Phi-3  
[√] Lama3_1  
[√] Gemma2   
[ ] Mini-CPM  
[ ] Falcon  
[ ] Qwen2  
[ ] ChatGLM  
Provides fine-tuning processes for open source large models to simplify the deployment, use, and application processes of open source large models.

For the web demo built with streamlit, you can run the program directly using the following command.

![webdemo](docs/webdemo.png)

```
streamlit run examples/xxx/xxx-web-demo.py --server.address 127.0.0.1 --server.port 8080
```

You can also try to use simple script to test local model update with latest weight file from server side.
You only need to run the restful api script and then run the client script.

Open one terminal as server side and run:
```commandline
# run api script so that client side can fetch model weight file from server side
python run_fast_api.py
```

Open another terminal as client side and run:
```commandline
# run client script and set local_update to True to simulate the update of the local model
python run_grpc_client.py --local_update true
```

> Make sure you server side already aggregate and [save](#server_side_weight_save) a new weight file.
> 
> You can also save new model weights manually by coping the weight file from server side to your client side. **You need to copy both version folder and bin file from the server side.** For example:
> 
> ```
> - client
>   |_ weights_update
>       |_ 20240829_162336
>           |_ adapter_model.bin
> ```
> 
> The **weight_file_download_path** in `config.yaml` is the path to save model weight from server side, make sure you put the weight file to this folder if you manually copy file from the server.
> ```
> client:
>  weight_file_download_path: "./client/weights_update" 
>  auto_pull: True # set it to False if you want to copy weight file from server side manually. If it's False, make sure you already put the weight file to the right place before you call update function in client side.
>```

You can check the details in `client/client.py`:
```python
def update(self):
    if self.config_detail.client.auto_pull is True:
        try:
            url = f'{self.config_detail.server.restful_url}/latest_weight'
            response = requests.get(url, stream=True)
            response.raise_for_status()
            content_disposition = response.headers.get('Content-Disposition', '')
            filename = content_disposition.split('filename=')[-1]
            model_version = filename.split('-')[0]
            save_path = os.path.join(self.model_weights_download_path, model_version)
            os.makedirs(save_path, exist_ok=True)
            with open(save_path + '/adapter_model.bin', 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"Saved in {save_path}")
        except requests.RequestException as e:
            print(f"Download weight file failed, please download manually. Error: {e}")
            raise requests.RequestException(f"Download weight file failed, please download manually. Error: {e}")

    lora_config_path = os.path.join(self.config_detail.sft.training_arguments.output_dir,
                                    'adapter_config.json')
    _, lora_weights_path_list = get_latest_folder(self.model_weights_download_path)
    if os.path.isfile(lora_config_path) and lora_weights_path_list:
        lora_weights_path = os.path.join(lora_weights_path_list[0],
                                         'adapter_model.bin')
        config = LoraConfig.from_pretrained(self.config_detail.sft.training_arguments.output_dir)
        lora_weights = torch.load(lora_weights_path)
        model = PeftModel(self.model, config)
        set_peft_model_state_dict(model, lora_weights)
        self.model = model
    else:
        print("No Lora config and weights found. Skipping update.")
```

## Blockchain Support

In order to ensure fairness and security, we have designed a decentralized on-chain role-based access control architecture. Different roles have different permissions and functions to ensure that user data is secure and cannot be tampered with on the chain. For relevant details, please visit [FLBRoleDesign](./chain/FLBRoleDesign.md) document.

**Privacy**: We develop smart contract on chain to support data record and reward distribution in blockchain.

We have deployed a Solidity smart contract on the Cerbo Chain to record user training data and user scores. The files related to the smart contract project are located in the `chain/contract/` directory, developed based on the Hardhat framework, an EVM-based smart contract development environment. Developers can also refer to this contract code to deploy their contracts. For more detailed information, please refer to the smart contract project's [README-Chain](../chain/README.md) document.

```solidity
contract FLB is Ownable {
    using EnumerableMap for EnumerableMap.AddressToUintMap;

    EnumerableMap.AddressToUintMap private scores;   // storing user scores
    mapping(string => bool) private dataUploaded;    // record data is uploaded
    mapping(address => string[]) public userDataSet; // storing user upload data

    // Event, used to record score add
    /**
     * @dev ScoreAdd defines an Event emitted when admin update user's score.
     */
    event ScoreAdd(address indexed user, uint256 newScore);
    /**
     * @dev DataUpload defines an Event emitted when user updaload his local data for train.
     */
    event DataUpload(address indexed user);

    constructor(address initialOwner) Ownable(initialOwner) {}

    /**
     * @dev addScore defines admin update user's score.
     */
    function addScore(address user, uint256 score) public onlyOwner {
        (bool find, uint256 preScore) = scores.tryGet(user);
        scores.set(user, find ? preScore + score : score);
        emit ScoreAdd(user, score); // Triggering Events
    }

    /**
     * @dev uploadUserData defines user updaload his local data for train.
     */
    function uploadUserData(string memory data) public {
        require(!dataUploaded[data], "Data already uploaded");

        userDataSet[msg.sender].push(data);
        dataUploaded[data] = true;

        if (!scores.contains(msg.sender)) {
            scores.set(msg.sender, 0);
        }

        emit DataUpload(msg.sender);
    }
}
```

### Reward Score Calculation Formula
Right now we use **number of times to upload weights * size of weight file uploaded each time + total training data volume** as the main factor rewards for each client.

#### TODO 
- [x] Score = number of times to upload weights * size of weight file uploaded each time + total training data volume
- [x] base_factor = number of times to upload weights * size of weight file uploaded each time + total training data volume <br> coefficient = (New_Performance−Original_Performance) / Original_Performance <br> Score = base_factor * coefficient
> The base factor calculation is in `server/calculate.py`:
>```python
> def calculate_client_scores(clients_uploads, coefficient, weight_size=2, total_rewards=100):
>    client_scores = {}
>
>    for client_id, uploads in clients_uploads.items():
>        upload_times = len(uploads)
>        total_train_dataset_length = sum([upload[1] for upload in uploads])
>
>        # Calculate the score based on the formula: upload times * weight_size + total train dataset length
>        score = upload_times * weight_size + total_train_dataset_length
>        client_scores[client_id] = score
>
>    # Calculate the total score
>    total_score = sum(client_scores.values())
>
>    # Calculate the percentage for each client
>    client_score_percentages = {client_id: round(score / total_score, 4) * total_rewards * coefficient for client_id, score in
>                                client_scores.items()}
>
>    return client_score_percentages
> ```
> The **coefficient** is calculated in `server/server.py`:
> ```python
>    def calculate_reward(self, clients):
>        if self.latest_version != self.pre_version:
>            latest_result_path = os.path.join(self.output, self.latest_version, 'eval_result.json')
>            pre_result_path = os.path.join(self.output, self.pre_version, 'eval_result.json')
>            with open(latest_result_path, 'r') as f:
>                latest_result = json.load(f)
>            if os.path.isfile(pre_result_path):
>                with open(pre_result_path, 'r') as f:
>                    pre_result = json.load(f)
>            else:
>                pre_result = eval_model(self.config_detail.model.model_path, n_train=1)
>                with open(pre_result_path, 'w') as f:
>                    json.dump(pre_result, f)
>                print(f'Pre version {self.pre_version} eval results: {pre_result}')
>            coefficient = min(0, latest_result['total']['accu'] - pre_result['total']['accu']) / latest_result['total']['accu'] if latest_result['total']['accu'] > 0 else 0
>        else:
>            coefficient = 1
>        score_percentage = calculate_client_scores(clients, coefficient)
>        for user_address, score in score_percentage.items():
>            send_score(user_address, score)
>```


> You can set **chain_record** in `config.yaml` to **True** if you want to connect with blockchain. The default value of **chain_record** is **False**.

In order to interact with the blockchain, you need to prepare `node.js v20.16.0` or higher.

### Creating Accounts

Two roles are required in the interaction with the chain: administrator and user. The administrator is the creator of the smart contract and is responsible for updating the user score to the chain; the user as a client is a participant of the blockchain and can upload model weight to the chain. Generate administrator and user accounts by executing the scripts `createAdmin.js` and `createAccount.js`:

```shell
cd ./chain/contract/sample
# Install Dependencies
npm install
node createAdmin.js

node createAccount.js
```

After execution, the account information of the administrator and user, including the private key and address, will be printed on the console. You need to copy the private key and address to the corresponding location of the `chain/config.json` file.

For example, the printed account information might look like this:

```text
admin account privateKey: 0xb3fb07bbe4570033909abe7a21dd6f28446c52ccd0cfe2c1d6caad4fdaf8e2b3
admin account address: 0x50864136432D42C940971c6c4A5B9318638F6Bb0
```
```text
user account privateKey: 0x93a94ad51cde0f31bcd264491fbf1195573a74126a9d10c180d54a7af3bae58a
user account address: 0xA178d222D1a5B30900A3DFC404876cf8340048C9
```

> When you need to run a real environment with multiple devices (multiple clients), you can generate multiple users by executing the script `createAccount.js` multiple times, and configure the generated user address and private key to the `chain/config.json` of the corresponding **client** end.

## Tokens

### Token Usage

Interacting with the chain requires a certain amount of gas, which is common practice in blockchain to prevent malicious users from consuming a large amount of on-chain resources and causing network congestion. Therefore, you need to acquire tokens to pay for the transaction gas fees. In the Cerbo Chain, we use `CBO` tokens as gas fees.

In the process of interacting with the chain, a certain amount of gas needs to be consumed, so you need to transfer some tokens to the administrator account and the ordinary user account respectively. Get test tokens through this faucet link: https://dev-faucet.cerboai.com/ Just pass in the address and click the button.

### Deploy the contract

Deploying the contract requires the administrator account and some other configuration items. Deploy the contract by running the following command:

```
cd ./chain/contract/sample
node deployContract.js
```

After the execution is completed, the contract address will be printed out in the console. You need to copy the contract address to the corresponding location of the `chain/config.json` file.

For example, the printed contract address might look like this:

```text
contract address: 0xF872b3704980BC87e87E5EabDFfCf357CBC28C0F
```

### Blockchain Explorer

During interaction with the chain, you can view transaction information through a blockchain explorer. The blockchain explorer address is https://dev-scan.cerboai.com/. For example, the contract deployment transaction mentioned above can be seen in detail on the blockchain explorer, including the transaction hash, transaction status, transaction fee, and timestamp.

![be.png](./docs/be.png)

## Additional Information

For more details on the project's implementation and usage, please refer to the documentation provided in the repository.

This README serves as a quick start guide to set up and run the integration of Federated Learning and blockchain technology. For any issues or contributions, please refer to the repository's issue tracker and contribution guidelines.

Under Construction .. 
