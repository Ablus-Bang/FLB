Language: [[English]](./README.md) [[中文]](./README-cn.md)

### Overview
This is a smart contract project that records compress user train data and user scores 

Run this project you should hava node.js environment `v20.x`

### Smart Contract Test
This project is based on Hardhat, a development environment for Ethereum smart contracts.

1. Install NPM

`npm install
`

2. Compile the smart contract

`npx hardhat compile`

3. Start a localhost network

`npx hardhat node`

4. Open a new terminal and deploy the smart contract in the localhost network

`npx hardhat run --network localhost scripts/deployFLB.js`

5. The following content is displayed in the terminal, indicating that the contract has been deployed successfully

```shell
Deploying contracts with the account: 0x00000Be6819f41400225702D32d3dd23663Dd690
Account balance: 100000000000000000000000000000000000000000000000000000
flb address: 0x546bc6E008689577C69C42b9C1f6b4C923f59B5d
0x00000Be6819f41400225702D32d3dd23663Dd690 have score 100
```

6. Run the test

`npx hardhat test`

To run a specified test file, you can use the following command:

`npx hardhat test ./test/flb.js`

7. The following information is displayed in the terminal, indicating that the contract has been tested successfully:

```shell
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


### Run the Sample

`cd ./sample`

1. Install the dependencies

`npm install`

2. Modify the configuration file if necessary
`config.json`

3. Run the sample

`node flb.js`


### Verify contract code

`npx hardhat verify --network cerbo 0xfF45Ac560476aEd6F7794f0e835b61d95e5d1C21 "0x00000Be6819f41400225702D32d3dd23663Dd690" --show-stack-traces --force`

### See on blockscout

https://dev-scan.cerboai.com/address/0xfF45Ac560476aEd6F7794f0e835b61d95e5d1C21?tab=txs