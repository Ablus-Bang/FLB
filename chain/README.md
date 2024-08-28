Language: [[English]](./README.md) [[中文]](../docs/README-Chain-cn.md)

### Overview
This is a smart contract project developed in the [solidity](https://soliditylang.org/) language, which is used to record user training data and user scores.
The blockchain platform used is Cerbo Chain, which is compatible with the EVM virtual machine.

To run this project, you need [Node.js](https://nodejs.org/en) v20.16.0 and above


### Contract Code Analysis
Based on the Hardhat framework, we have developed a smart contract that implements the functionality to record user training data and user scores, located in the `contracts/FLB.sol` file.

#### Contract Basic Structure:
The FLB contract inherits from OpenZeppelin's Ownable library, which means that the contract owner (administrator) has special permissions.

#### Data Storage:
The EnumerableMap is used to store user scores, allowing for efficient insertion, deletion, and lookup operations.  
The dataUploaded mapping is used to record whether a certain piece of data has been uploaded, preventing duplicate uploads.  
The userDataSet array stores the data uploaded by each user for easy retrieval later.

#### Events:
The ScoreAdd event is triggered when the administrator updates a user's score, recording relevant information.  
The DataUpload event is triggered when a user uploads data, recording the address of the uploader.

#### Constructor:
The constructor accepts an address as the initial administrator and initializes it through Ownable. This allows the contract owner to manage user scores.

#### Core Functions:
addScore: This method can only be called by the administrator, allowing them to increase the score for a specified user. The current score is retrieved using the tryGet method; if the user does not exist, it is initialized to 0.  
uploadUserData: Users can upload data, but must ensure that the data has not been uploaded before. If the data upload is successful, the user's score will be initialized to 0 (if it does not already exist).  
userExists: Checks if a user exists in the system.  
getScore: Returns the score of the specified user; if the user does not exist, it returns 0.  
getUserData and getUserDataCount: Provide retrieval functions for user-uploaded data.  
getUserCount and getUserScore: Used to obtain the current number of users and to get user scores by index.

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

### Smart Contract Test
Because this project is developed based on the Hardhat framework, contract testing can be performed locally,
mainly to verify whether the contract functions normally.

`cd ./contract/`

1. Install NPM

`npm install
`

2. Compile the smart contract

`npx hardhat compile`

3. Start a localhost network  
   Running this command will start a single-node blockchain network locally for deploying and testing contracts.

`npx hardhat node`

4. Open a new terminal and deploy the smart contract to the local network through the automatically generated administrator account.

`npx hardhat run --network localhost scripts/deployFLB.js`

5. The terminal displays the following content, indicating that the contract has been successfully deployed.

```text
Deploying contracts with the account: 0x00000Be6819f41400225702D32d3dd23663Dd690
Account balance: 100000000000000000000000000000000000000000000000000000
flb address: 0x546bc6E008689577C69C42b9C1f6b4C923f59B5d
0x00000Be6819f41400225702D32d3dd23663Dd690 have score 100
```

6. Run the test  
   Run the following command to test the contract functionality using the contract that has been deployed on the local blockchain network.

`npx hardhat test`

To run a specified test file, you can use the following command:

`npx hardhat test ./test/flb.js`

7. The terminal displays the following information, indicating that the contract has been tested successfully.
   From the printed information, it can be seen that the administrator calls the contract's `addScore` method,
   throws a contract event, and non-administrators cannot call the `addScore` method. These test cases have all passed.

```text
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
Here we have prepared an example. The FLB contract has been deployed in Cerbo Chain.
You can run the example with the following command. In this example, we called the contract's `addScore`
method to add a score to the contract, called the `uploadUserData` method to upload local data,
and the corresponding query method.

`cd ./contract/sample`

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