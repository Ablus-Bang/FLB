const { ethers } = require('hardhat');

async function main() {
  const [deployer] = await ethers.getSigners();
  let tx;

  console.log('Deploying contracts with the account:', deployer.address);

  // Get balance from Hardhat instance using ethers.js
  const balance = await deployer.provider.getBalance(deployer.address);
  console.log('Account balance:', balance.toString());

  const FLB = await ethers.getContractFactory('FLB');
  const flb = await FLB.deploy(deployer.address);

  await flb.waitForDeployment();

  console.log('flb address:', await flb.getAddress());

  tx = await flb.addScore(deployer.address, 100);
  await tx.wait();

  const score = await flb.getScore(deployer.address);
  console.log(`${deployer.address} have score ${score}`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
