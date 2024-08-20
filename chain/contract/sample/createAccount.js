import {ethers} from "ethers";

export const createAccount = () => {
    const userAccount = ethers.Wallet.createRandom();
    console.log("user account privateKey:", userAccount.privateKey);
    console.log("user account address:", userAccount.address);
}

createAccount()