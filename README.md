# Deep Q-Network (DQN) Breakout Experiments

This repository contains various implementations and experiments with Deep Q-Networks (DQN). Each folder contains a specific variant or extension of the standard DQN algorithm.

To get started, follow the instructions in requirements.txt to construct a virtual environment.
Run the environment_test.py and you are expected to see no importing errors and a pop up window playing the game Breakout.

Start with dqn_all.py under the folder DQN.

---

## Folder Descriptions

### `DQN`
This folder contains the baseline implementation of the standard Deep Q-Network (DQN) algorithm. It serves as a reference implementation.

### `DQN_More_Replay`
This variant has larger replay buffer.

### `DQN_No_Frame_Selection`
This variant remove frame selection.

### `DQN_Slow`
A variant of DQN with large exploration_steps=1_000_000, replay_start_size=50_000, device='cuda', C=10_000.

### `DQN_no_target_network_with_ExpReplay`
An experimental version that removes the target network but still utilizes experience replay for training. 

### `DQN_punish`
Introduces a custom punishment mechanism in the reward function to discourage losing life.
 
```bash 
