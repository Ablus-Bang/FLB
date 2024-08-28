# **Role-Based Access Control (RBAC)**
Language: [[English]](./FLBRoleDesign.md) [[中文]](../docs/FLBRoleDesign-cn.md)


**Overview**

Access control—"who is allowed to do this"—is crucial in building our ecosystem. For example, access control might determine who can record scores for users. The most common and basic form of access control is the concept of ownership: there is an account in the system designated as the `owner`, which can perform administrative tasks.

While the simplicity of ownership is useful for simple systems or rapid prototyping, different levels of authorization are often needed. For instance, one account might be able to ban users from accessing the system but not create new tokens. Role-Based Access Control (RBAC) provides flexibility in this regard.

Essentially, we will define multiple roles, each allowing a different set of operations to be performed, instead of having `onlyOwner` everywhere.

We manage these roles through voting governance based on actual business scenarios. Specific information is explained in the Voting Governance section.

**Roles**

The following four roles exist in our ecosystem:

- Uploader: Responsible for uploading relevant data to the blockchain system for model training.
- Verifier: Data validator
    - Responsibilities: Verifiers are responsible for checking whether the parameters uploaded by participants meet quality requirements and independently validating these parameters.
    - Functions:
    - Validate contributions: Verifiers evaluate participants' contributions, determining whether they are valid (e.g., if they improve the performance of the global model).
- Challenger: To prevent Uploaders from uploading invalid data, Challengers verify the data uploaded by Uploaders.
    - Responsibilities: Challengers can challenge participant contributions that have been approved by verifiers and conduct independent verification. If the challenge is successful, the challenger receives a reward, and the participant is penalized.
- Finalizer: Makes the final decision on the results of challenges by Challengers.

## Implementation

For each defined role type (Uploader, Verifier, Challenger, Finalizer), we define a variable of type Role that uses a map to store the list of accounts with that role.

The related pseudocode is shown below:

```go
enum RoleType (
  Uploader = keeack256("UPLOADER");
  Verifier = keeack256("VERIFIER");
  Challenger = keeack256("CHALLENGER");
  Finalizer = keeack256("FINALIZER");
)

struct Role {
	hasRole map[address]bool;
}

roles := map[RoleType]Role

func HasRole(RoleType roleType, address account) (bool) {
    return roles[roleType].hasRole[account];
}

func GrantRole(RoleType roleType, address account) {
    roles[roleType].hasRole[account] = true;
}

func RevokeRole(RoleType roleType, address account) {
    roles[roleType].hasRole[account] = false;
}

```

From the design above, we can see that an account can actually have multiple roles. This flexibility can be applied in practice: for example, in some business scenarios, an account may need multiple roles to succeed.

**Role Granting/Revoking**

We could have an Owner role for granting and revoking the above roles. However, this brings the danger of centralization. If the Owner's private key is leaked or lost, it would have an immeasurable impact on the entire system. Therefore, we can leverage the gov module of Cosmos, using the form of initiating proposals to grant/revoke roles, achieving the goal of decentralization.

Taking granting/revoking an Uploader role as an example, the related pseudocode is as follows:

```go
// GrantUploader defines a method for grant a new uploader
func (k msgServer) GrantUploader(ctx context.Context, msg *types.MsgGrantUploader) (*types.MsgGrantUploaderResponse, error) {
	signer := msg.GetSigner()
	if !signer.Equals(k.accountKeeper.GetModuleAddress(govtypes.ModuleName)) {
		return nil, types.ErrSignerNotGovModule
	} else {
		GrantRole(RoleTypeUploader, msg.uploader);
		k.bankKeeper.SendCoinsFromAccountToModule(ctx, msg.uploader, types.ModuleName, coins)
	}
}

// RevokeUploader defines a method for revoke a uploader
func (k msgServer) RevokeUploader(ctx context.Context, msg *types.MsgRevokeUploader) (*types.MsgRevokeUploaderResponse, error) {
	signer := msg.GetSigner()
	if !signer.Equals(k.accountKeeper.GetModuleAddress(govtypes.ModuleName)) {
		return nil, types.ErrSignerNotGovModule
	} else {
		RevokeRole(RoleTypeUploader, msg.uploader);
		k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, msg.uploader, coins)
	}
}
```

To prevent Uploaders from maliciously uploading invalid data, when granting an account the Uploader role, we need to transfer some of the account's tokens to the module account for staking. If a Challenger role successfully challenges the data, part of the Uploader's staked tokens need to be deducted to reward the Challenger. Once all staked tokens are deducted, the system automatically revokes the user's Uploader role. At this point, the user needs to replenish the minimum required amount of staked tokens to restore their Uploader role.

Similarly, if a user's Uploader role is revoked, we need to return the user's staked tokens from the system to the user.

**Invocation**

- **Internal Module Invocation**

Role is a basic module. The purpose of building this module is to provide access management control for other modules to call. The entire functionality is basically implemented in the Keeper. `Keeper` refers to the Cosmos SDK abstraction, which manages access to the state subset defined by each module. `Keeper`s are module-specific, meaning that the state subset defined by a module can only be accessed by the `keeper` defined in that module. Suppose the AI module needs to access the state subset defined by the Role module, then an internal `keeper` reference to the Role module needs to be passed to the AI module. The related AI pseudocode is as follows:

```go
// RoleKeeper defines the expected interface needed to check account role.
type RoleKeeper interface {
	func HasRole(RoleType roleType, address account) (bool)
}

// UploadData defines a method a uploader updalod data
func (k msgServer) UploadData(ctx context.Context, msg *types.MsgUploadData) (*types.MsgUploadDataResponse, error) {
	signer := msg.GetSigner()
	if !k.roleKeeper.HasRole(signer) {
		return nil, types.ErrSignerNotHasUploaderRole
	} else {
		SaveData(msg.data);
	}
}

```

- Smart Contract Invocation

Since our chain supports Solidity contracts, to expand use cases and allow Solidity contracts to also have access to the Go-implemented role module, we can implement the role module using precompiled contracts. Suppose the address of our implemented precompiled contract is:
`0x0000000000000000000000000000000000006000`

Below is an example of how a contract can call this precompiled contract to access the underlying Role module:
```solidity
pragma solidity ^0.8.0;

/// @dev The IRole contract's address.
address constant ROLE_PRECOMPILE_ADDRESS = 0x0000000000000000000000000000000000006000;

/// @dev The IRole contract's instance.
IRole constant ROLE_CONTRACT = IRole(ROLE_PRECOMPILE_ADDRESS);

interface IRole {
    /**
     * @dev Returns `true` if `account` has been granted `role`.
     */
    function hasRole(bytes32 role, address account) external returns (bool);
}

contract RoleTest {
    function testHasRole(bytes32 role, address account) public returns (bool) {
        return ROLE_CONTRACT.hasRole(role, account);
    }
}
```

## Cerbo Chain Architecture Design

![image.png](../docs/chain-architecture.png)

The overall structure of Cerbo Chain is divided into two main layers: the underlying Tendermint Core and the upper Application Layer, with specific details as follows:

**1. Tendermint Core**

- **Consensus Layer**: Responsible for implementing the Proof of Stake (PoS) consensus mechanism, ensuring nodes in the network reach agreement, guaranteeing the security and order of transactions.
- **Network Layer**: Responsible for communication between nodes, ensuring effective data transmission in the network. This layer provides peer-to-peer network protocols, supporting message broadcasting and synchronization.

**2. Application Layer**

- **Cosmos SDK**: This is the core framework of Cerbo Chain, providing basic modules for building blockchain applications. Specific modules include:
    - **Auth**: Responsible for on-chain account management, supporting account creation, updating, deletion, and other operations.
    - **Bank**: Manages asset transfers.
    - **Crisis**: Consistency check of on-chain states.
    - **Distribution**: Manages reward and distribution mechanisms.
    - **Gov**: Governance module, allowing community participation in decision-making.
    - **Staking**: Handles staking and validator management.
    - **Params**: Manages chain parameter settings.
    - **Slashing**: Manages passive malicious behavior of validators.
    - **Evidence**: Manages active malicious behavior of validators.
    - **Genutil**: Tool for generating genesis blocks.
    - **IBC (Inter-Blockchain Communication)**: Supports cross-chain communication, allowing interaction between different blockchains.
    - **Others**: Other possible modules.
- **Extensions**: These modules provide additional functionality, enhancing the capabilities of Cerbo Chain:
    - **Inflation**: Manages inflation and token issuance.
    - **ERC20**: Supports compatibility with the Ethereum ERC20 token standard.
    - **Role**: Defines and manages different roles on the chain, including Uploader, Verifier,Challenger, and Finalizer.
    - **Action**: Handles specific operations. For example, uploading data, updating scores, etc., these operations are completed by the Uploader role.
    - **EVM (Ethereum Virtual Machine)**: Allows running Ethereum smart contracts on Cerbo Chain, enhancing compatibility with the Ethereum ecosystem.

## **Cerbo Chain Voting Governance**

In the Cerbo Chain community, anyone can submit a proposal. After submission, the community follows a strict governance process to decide whether the proposal should be implemented.

1. **Deposit Phase**

After a proposal is submitted, it needs to receive a deposit of at least 512 (tentative) CBO within two weeks. Since anyone in the community can submit a proposal, the 512 CBO is the minimum threshold for a proposal to enter the voting phase, used to eliminate costless spam proposals.

2. **Voting Phase**

After reaching the minimum deposit of 512 CBO, the proposal will enter a two-week voting phase. All token stakers have voting rights: "Yes", "No", "NoWithVeto", "Abstain".

Note: Validators can vote on behalf of delegators who have staked tokens to them, but when delegators vote themselves, it will override the validator's vote.

3. **Tallying Votes**

A proposal needs to meet all three of the following conditions to enter the implementation phase:

1. After two weeks, more than 40% of the staked tokens in the system have participated in voting on the proposal.
2. More than 50% of the participating tokens (excluding "Abstain" votes) voted "Yes".
3. Less than 33.4% of the participating tokens (excluding "Abstain" votes) voted "NoWithVeto".

If any of the above three conditions are not met, the proposal will be rejected, and the deposited funds will not be refunded but will be deposited into the community pool.

4. **Implementing the Proposal**

After the community decides to accept the proposal through voting, it will enter the proposal execution phase.

![image.png](../docs/vote.png)