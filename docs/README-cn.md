# FLB - Project on the Integration of Federated Learning and Blockchain 

Language: [[English]](README.md) [[中文]](docs/README-cn.md)

## 介绍

本项目旨在将联邦学习 (FL) 与区块链技术相结合，增强机器学习应用中的数据隐私和安全性。联邦学习支持在持有本地数据样本的去中心化设备之间训练全局模型，而无需交换它们。通过结合区块链，我们可以确保联邦学习过程的环境不可篡改、透明且安全。

## 环境设置
克隆项目，准备本地环境：
```
conda create -n flb python=3.11
conda activate flb
pip install -r requirements.txt
```

## 设置配置文件
在我们最初的 **config.yaml** 文件中，你需要设置本地模型路径和 dataset_name。 <br>
> 目前我们仅支持 CUDA 进行训练，如果要使用 cpu，请在 `config.yaml` 中将 **quantization** 设置为 **null** 并将 **device_map** 设置为 **cpu**。

## 联邦学习微调

运行以下命令来启动本地联邦学习微调测试：

```
python main_simple_fl.py
```

> 客户端默认使用本地差分隐私，可以通过在 `config.yaml` 中将客户端块中的 **local_dp** 设置为 **False** 来关闭它
>
> 当前在服务器端，在 `server/strategy/` 下只设置了两个策略。默认策略是联邦平均。如果想在服务器端使用差分隐私，可以在`config.yaml`中将客户端块中的**local_dp**设置为**False**，并在`main_simple_fl.py`中给server设置dp策略

## 更多的开源模型微调集部署案例

我们将逐渐提供更多的开源大模型的微调教程，简化开源大模型的部署、使用、应用流程。 

对于使用streamlit搭建的web demo，可以使用以下命令直接运行程序

```
streamlit run examples/xxx/xxx-web-demo.py --server.address 127.0.0.1 --server.port 8080
```

## 支持区块链

我们部署了一个合约项目来支持区块链中的数据记录和奖励分配。

目前我们使用**上传权重的次数*每次上传的权重文件大小+总训练数据量**作为每个客户端的主要奖励因素。

> 如果不想连接区块链，可以在`config.yaml`中将**chain_record**设置为**False**。**chain_record**的默认值为**False**。

为了实现和区块链进行交互，首先需要准备`node.js v20.0.x`环境

### 创建账户
在和链的交互中需要两种角色：管理员和普通用户。管理员是智能合约的创建者，负责更新用户score到链上；普通用户是区块链的参与者，可以上传data到链上。通过执行脚本`createAdmin.js`和`createAccount.js`来生成管理员和普通用户的账户：
```shell
cd ./chain/contract/sample
# Install Dependencies
npm install
node createAdmin.js

node createAccount.js
```
执行完成后，会在控制台上打印出管理员和普通用户的账户信息，包括私钥和地址。你需要拷贝私钥和地址到`chain/config.json`文件对应的位置中。

> 当你需要运行多台设备（多个用户）的真实环境时，可以通过执行脚本`createAccount.js`多次来生成多个用户，并将生成的用户地址和私钥配置到相应**client**端的`chain/config.json`中。

### 领取测试代币
在和链交互的过程中，需要消耗一定的gas，所以需要分别给管理员账户和普通用户账号转入一点代币。通过这个水龙头链接领取测试代币：https://dev-faucet.cerboai.com/ 只需要传入地址，点击按钮即可。


### 部署合约
通过执行以下命令部署合约：
```shell
cd ./chain/contract/sample
node deployContract.js
```
执行完成后，会在控制台打印出合约地址。你需要拷贝合约地址到`chain/config.json`文件对应位置中。


## 附加信息

有关该项目实施和使用的更多详细信息，请参阅存项目中提供的文档。

本 README 可作为设置和运行联邦学习与区块链技术集成的快速入门指南。对于任何问题或贡献，请参阅存储库的问题跟踪器和贡献指南。

持续更新中..