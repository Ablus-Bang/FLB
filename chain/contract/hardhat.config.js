require('@nomicfoundation/hardhat-toolbox');
require('@nomicfoundation/hardhat-verify');

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    compilers: [
      {
        version: '0.8.24',
        settings: {
          optimizer: {
            enabled: true,
            runs: 200,
          },
        },
      },
    ],
  },
  defaultNetwork: 'localhost', // hardhat localhost
  networks: {
    hardhat: {
      accounts: [
        {
          privateKey: 'f78a036930ce63791ea6ea20072986d8c3f16a6811f6a2583b0787c45086f769',
          balance: '100000000000000000000000000000000000000000000000000000',
        },
        {
          privateKey: '95e06fa1a8411d7f6693f486f0f450b122c58feadbcee43fbd02e13da59395d5',
          balance: '100000000000000000000000000000000000000000000000000000',
        },
      ],
    },
    localhost: {
      url: 'http://127.0.0.1:8545',
      gasPrice: 10000000000,
      accounts: [
        'f78a036930ce63791ea6ea20072986d8c3f16a6811f6a2583b0787c45086f769',
        '95e06fa1a8411d7f6693f486f0f450b122c58feadbcee43fbd02e13da59395d5',
      ],
    },
    cerbo: {
      url: 'https://dev-rpc.cerboai.com',
      gasPrice: 10000000000,
      accounts: [
        'f78a036930ce63791ea6ea20072986d8c3f16a6811f6a2583b0787c45086f769',
        '95e06fa1a8411d7f6693f486f0f450b122c58feadbcee43fbd02e13da59395d5',
      ],
    },
  },
  etherscan: {
    apiKey: {
      localhost: 'It seems like you can write whatever you want',
      cerbo: 'It seems like you can write whatever you want',
    },
    customChains: [
      {
        network: 'localhost',
        chainId: 8888888,
        urls: {
          apiURL: 'http://127.0.0.1:4000/api',
          browserURL: 'http://127.0.0.1:4000',
        },
      },
      {
        network: 'cerbo',
        chainId: 8555,
        urls: {
          apiURL: 'https://dev-scan.cerboai.com/api',
          browserURL: 'https://dev-scan.cerboai.com/',
        },
      },
    ],
  },
  sourcify: {
    enabled: true,
  },
};
