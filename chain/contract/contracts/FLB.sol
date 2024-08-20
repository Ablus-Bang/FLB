// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/structs/EnumerableMap.sol";

contract FLB is Ownable {
    using EnumerableMap for EnumerableMap.AddressToUintMap;

    EnumerableMap.AddressToUintMap private scores; // storing user scores
    mapping(string => bool) private dataUploaded; // record data is uploaded
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

    /**
     * @dev userExists defines check user is exists in system.
     */
    function userExists(address user) external view returns (bool) {
        return scores.contains(user);
    }

    /**
     * @dev getScore defines get the score from records.
     */
    function getScore(address user) public view returns (uint256) {
        (bool find, uint256 score) = scores.tryGet(user);
        return find ? score : 0; // Returns the user's score
    }

    /**
     * @dev getUserData defines get the user's data by index.
     */
    function getUserData(
        address user,
        uint256 index
    ) public view returns (string memory) {
        require(index < userDataSet[user].length, "Index out of bounds");

        return userDataSet[user][index];
    }

    /**
     * @dev getUserDataCount defines get the user's data count.
     */
    function getUserDataCount(address user) public view returns (uint256) {
        return userDataSet[user].length;
    }

    /**
     * @dev getUserCount defines get the user's count.
     */
    function getUserCount() external view returns (uint) {
        return scores.length();
    }

    /**
     * @dev getUserScore defines get the user's score by index.
     */
    function getUserScore(uint index) external view returns (address, uint) {
        return scores.at(index);
    }
}
