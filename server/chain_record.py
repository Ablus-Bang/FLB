from web3 import Web3
import json


def send_score(user_address, score):
    with open('../chain_config/config.json') as config_file:
        config_data = json.load(config_file)

    rpc = config_data['url']
    contract_address = config_data['contract_address']
    private_key = config_data['private_key']

    # ABI of the contract
    with open('../chain_config/abi.json') as abi_file:
        abi = json.load(abi_file)

    # Initialize a Web3 instance
    web3 = Web3(Web3.HTTPProvider(rpc))

    # Ensure the connection is successful
    if not web3.is_connected():
        raise Exception("Failed to connect to Ethereum node")

    # Load the account using the private key
    account = web3.eth.account.from_key(private_key)
    score_contract = web3.eth.contract(address=contract_address, abi=abi)

    # Add score to an address
    # user_address = '0x2AD60507F1596Af20D5083Bf4baD4969643980e2'

    # Create a transaction to call the addScore function
    transaction = score_contract.functions.addScore(user_address, score).build_transaction({
        'from': account.address,
        'nonce': web3.eth.get_transaction_count(account.address)
    })

    signed_txn = web3.eth.account.sign_transaction(transaction, private_key)
    txn_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
    txn_receipt = web3.eth.wait_for_transaction_receipt(txn_hash)
    print('addScore success, receipt: ', txn_receipt)

    # Get the score for the address
    # score = score_contract.functions.getScore(user_address).call()
    # print('getScore success, score: ', score)
