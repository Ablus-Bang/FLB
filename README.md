# FLB


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
```
python main_simple_fl.py
```
