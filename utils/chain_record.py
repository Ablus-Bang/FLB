from web3 import Web3
import json


def send_score(user_address, score):
    """Send score from server side to client chain"""
    with open("../chain/config.json", "r") as config_file:
        config_data = json.load(config_file)

    rpc = config_data["url"]
    contract_address = config_data["contract_address"]
    private_key = config_data["admin_private_key"]

    # ABI of the contract
    with open("../chain/abi.json", "r") as abi_file:
        abi = json.load(abi_file)

    # Initialize a Web3 instance
    web3 = Web3(Web3.HTTPProvider(rpc))

    # Ensure the connection is successful
    if not web3.is_connected():
        raise Exception("Failed to connect to Ethereum node")

    # Load the admin account using the private key
    account = web3.eth.account.from_key(private_key)
    score_contract = web3.eth.contract(address=contract_address, abi=abi)

    # Add score to an address
    old_score = score_contract.functions.getScore(user_address).call()
    print("get old score: ", old_score)

    # Create a transaction to call the addScore function
    transaction = score_contract.functions.addScore(
        user_address, score
    ).build_transaction(
        {
            "from": account.address,
            "nonce": web3.eth.get_transaction_count(account.address),
        }
    )

    signed_txn = web3.eth.account.sign_transaction(transaction, private_key)
    txn_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
    txn_receipt = web3.eth.wait_for_transaction_receipt(txn_hash)
    print("addScore success, receipt: ", txn_receipt)

    # Get the score for the address
    new_score = score_contract.functions.getScore(user_address).call()
    print("get new score: ", new_score)


def send_weight(weight_dict):
    """Record model weight from client side to chain"""
    with open("../chain/config.json", "r") as config_file:
        config_data = json.load(config_file)

    rpc = config_data["url"]
    contract_address = config_data["contract_address"]
    private_key = config_data["admin_private_key"]

    user_address = config_data["user_address"]
    user_private_key = config_data["user_private_key"]

    # ABI of the contract
    with open("../chain/abi.json", "r") as abi_file:
        abi = json.load(abi_file)

    # Initialize a Web3 instance
    web3 = Web3(Web3.HTTPProvider(rpc))
    # Ensure the connection is successful
    if not web3.is_connected():
        raise Exception("Failed to connect to Ethereum node")

    # TODO might remove token sending in public version
    # Load the admin account using the private key
    admin_account = web3.eth.account.from_key(private_key)
    weight_contract = web3.eth.contract(address=contract_address, abi=abi)
    # send token to client first
    token_send_signed_txn = web3.eth.account.sign_transaction(
        dict(
            nonce=web3.eth.get_transaction_count(admin_account.address),
            to=user_address,
            value=pow(10, 18),
            data=b"",
            gas=21000,
            gasPrice=web3.eth.gas_price,
            chainId=8555,
        ),
        private_key,
    )
    token_send_txn_hash = web3.eth.send_raw_transaction(
        token_send_signed_txn.rawTransaction
    )
    token_send_txn_receipt = web3.eth.wait_for_transaction_receipt(token_send_txn_hash)
    print("send token success, receipt: ", token_send_txn_receipt)

    user_account = web3.eth.account.from_key(user_private_key)
    # Create a transaction to call the uploadUserData function
    transaction = weight_contract.functions.uploadUserData(
        weight_dict
    ).build_transaction(
        {
            "from": user_account.address,
            "nonce": web3.eth.get_transaction_count(user_account.address),
        }
    )
    signed_txn = web3.eth.account.sign_transaction(transaction, user_private_key)
    txn_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
    txn_receipt = web3.eth.wait_for_transaction_receipt(txn_hash)
    print("record weight success, receipt: ", txn_receipt)

    # data_count = weight_contract.functions.getUserDataCount(user_account.address).call()
    # print('get user data count: ', data_count)
    # new_data = weight_contract.functions.getUserData(user_account.address, data_count - 1).call()
    # print('get latest data, data: ', new_data)
