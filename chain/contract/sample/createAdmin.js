import {ethers} from "ethers";

export const createAdmin = () => {
    const adminAccount = ethers.Wallet.createRandom();
    console.log("admin account privateKey:", adminAccount.privateKey);
    console.log("admin account address:", adminAccount.address);
}

createAdmin()