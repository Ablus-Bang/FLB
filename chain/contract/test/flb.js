const { expect } = require('chai');
const { ethers } = require('hardhat');

describe('FLB', function () {
  let FLB;
  let flb;
  let owner;
  let nonOwner;

  before(async function () {
    // Deploy a new contract before each test
    [owner, nonOwner] = await ethers.getSigners(); // Get default accounts
    FLB = await ethers.getContractFactory('FLB');
    flb = await FLB.deploy(owner.address);
    console.log('flb address:', flb.target);
  });

  it('Should add score record', async function () {
    // add a record
    await flb.addScore('0x00000Be6819f41400225702D32d3dd23663Dd690', 100);
    expect(await flb.getScore('0x00000Be6819f41400225702D32d3dd23663Dd690')).to.equal(100);
  });

  it('Should add score record and emit event', async function () {
    await expect(flb.addScore('0x4B20993Bc481177ec7E8f571ceCaE8A9e22C02db', 100))
      .to.emit(flb, 'ScoreAdd')
      .withArgs('0x4B20993Bc481177ec7E8f571ceCaE8A9e22C02db', 100);
  });

  it('Should accumulate score when adding multiple times', async function () {
    await flb.addScore('0x00000Be6819f41400225702D32d3dd23663Dd690', 100);
    expect(await flb.getScore('0x00000Be6819f41400225702D32d3dd23663Dd690')).to.equal(200);
  });

  it('Should not allow non-admin to add score', async function () {
    const nonOwnerAddress = nonOwner.address;

    await expect(flb.connect(nonOwner).addScore('0x1111102Dd32160B064F2A512CDEf74bFdB6a9F96', 100))
      .to.be.revertedWithCustomError(flb, 'OwnableUnauthorizedAccount')
      .withArgs(nonOwnerAddress);
  });
});
