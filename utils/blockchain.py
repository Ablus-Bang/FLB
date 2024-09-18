import hashlib
import json
import time


class Block:
    def __init__(self, index, transactions, timestamp, previous_hash):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]
        self.pending_transactions = []
        self.mining_reward = 10  # Reward for adding a block

    def create_genesis_block(self):
        return Block(0, [], time.time(), "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, block):
        block.previous_hash = self.get_latest_block().hash
        block.hash = block.calculate_hash()
        self.chain.append(block)

    def add_transaction(self, sender, recipient, amount):
        self.pending_transactions.append(
            {"sender": sender, "recipient": recipient, "amount": amount}
        )

    def mine_pending_transactions(self, miner_address):
        block = Block(
            len(self.chain),
            self.pending_transactions,
            time.time(),
            self.get_latest_block().hash,
        )
        self.add_block(block)

        self.pending_transactions = [
            {
                "sender": "MINING_REWARD",
                "recipient": miner_address,
                "amount": self.mining_reward,
            }
        ]

    def get_balance(self, address):
        balance = 0
        for block in self.chain:
            for transaction in block.transactions:
                if transaction["recipient"] == address:
                    balance += transaction["amount"]
                if transaction["sender"] == address:
                    balance -= transaction["amount"]
        return balance


# Global blockchain instance
federated_chain = Blockchain()
