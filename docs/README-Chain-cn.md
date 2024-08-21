Language: [[English]](./README.md) [[中文]](./README-cn.md)

### 概览
这是一个智能合约项目，用于记录压缩的用户训练数据和用户得分

运行该项目需要有 Node.js 环境 `v20.x`

### 智能合约测试

本项目基于 Hardhat，这是一个 Ethereum 智能合约的开发环境。

1. 安装 NPM

`npm install`

2. 编译智能合约

`npx hardhat compile`

3. 启动本地网络

`npx hardhat node`

4. 打开一个新终端并在本地网络中部署智能合约

`npx hardhat run --network localhost scripts/deployFLB.js`

5. 终端显示以下内容，表示合约已成功部署

```
Deploying contracts with the account: 0x00000Be6819f41400225702D32d3dd23663Dd690
Account balance: 100000000000000000000000000000000000000000000000000000
flb address: 0x546bc6E008689577C69C42b9C1f6b4C923f59B5d
0x00000Be6819f41400225702D32d3dd23663Dd690 have score 100

```

6. 运行测试

`npx hardhat test`

要运行特定的测试文件，可以使用以下命令：

`npx hardhat test ./test/flb.js`

7. 终端显示以下信息，表明合约已成功测试：

```
Deploying contracts with the account: 0x00000Be6819f41400225702D32d3dd23663Dd690
Account balance: 100000000000000000000000000000000000000000000000000000
flb address: 0x546bc6E008689577C69C42b9C1f6b4C923f59B5d
0x00000Be6819f41400225702D32d3dd23663Dd690 have score 100
➜  contract git:(main) ✗ npx hardhat test ./test/flb.js


  FLB
flb address: 0x33Add53fb1CDeF4A10BeE7249b66a685200DDd2f
    ✔ Should add score record
    ✔ Should add score record and emit event
    ✔ Should accumulate score when adding multiple times
    ✔ Should not allow non-admin to add score


  4 passing (139ms)

```

### 运行示例

`cd ./sample`

1. 安装依赖

`npm install`

2. 如有必要，修改配置文件
`config.json`
3. 运行示例

`node flb.js`

### 验证合约代码

`npx hardhat verify --network cerbo 0xfF45Ac560476aEd6F7794f0e835b61d95e5d1C21 0x00000Be6819f41400225702D32d3dd23663Dd690 --show-stack-traces --force`

### 在区块浏览器上查看

[https://dev-scan.cerboai.com/address/0xfF45Ac560476aEd6F7794f0e835b61d95e5d1C21?tab=txs](https://dev-scan.cerboai.com/address/0xfF45Ac560476aEd6F7794f0e835b61d95e5d1C21?tab=txs)