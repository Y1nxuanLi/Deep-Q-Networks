import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.buffers import ReplayBuffer
import ale_py
import gymnasium as gym
from gymnasium import spaces
import pickle


class DQN(nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, nb_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


def Deep_Q_Learning(env, buffer_size=1_000_000, nb_epochs=30_000_000, train_frequency=4, batch_size=32,
                    gamma=0.99, replay_start_size=50_000, epsilon_start=1, epsilon_end=0.1,
                    exploration_steps=1_000_000, device='cuda', learning_rate=1.25e-4):
    # Initialize replay buffer
    rb = ReplayBuffer(buffer_size, env.observation_space, env.action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    # Initialize Q-network
    q_network = DQN(env.action_space.n).to(device)

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)

    epoch = 0
    total_rewards_list = []
    smoothed_rewards = []
    rewards = []
    total_loss_list = []
    loss_means = []
    losses = []
    best_reward = 0

    progress_bar = tqdm(total=nb_epochs)
    while epoch <= nb_epochs:

        dead = False
        total_rewards = 0

        # Reset environment
        obs, _ = env.reset()

        for _ in range(random.randint(1, 30)):  # Noop and fire to reset environment
            obs, reward, terminated, truncated, info = env.step(1)

        while not dead:
            current_life = info['lives']

            epsilon = max((epsilon_end - epsilon_start) / exploration_steps * epoch + epsilon_start, epsilon_end)
            if random.random() < epsilon:  # With probability ε select a random action a
                action = np.array(env.action_space.sample())

            else:  # Otherwise select a = max_a Q(φ(st), a; θ)
                q_values = q_network(torch.Tensor(obs).unsqueeze(0).to(device))
                action = np.array(torch.argmax(q_values, dim=1).item())

            # Execute action a in emulator
            next_obs, reward, terminated, truncated, info = env.step(action)
            dead = terminated or truncated

            done = np.array(info['lives'] < current_life)

            # Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)  # Reward clipping

            # Store transition in replay buffer
            rb.add(obs, next_obs, action, reward, done, info)

            obs = next_obs

            # Train network
            if epoch > replay_start_size and epoch % train_frequency == 0:
                # Sample a minibatch of transitions
                data = rb.sample(batch_size)
                with torch.no_grad():
                    max_next_q_values, _ = q_network(data.next_observations).max(dim=1)
                    target_q_values = data.rewards.flatten() + gamma * max_next_q_values * (1 - data.dones.flatten())
                current_q_values = q_network(data.observations).gather(1, data.actions).squeeze()

                # Compute loss and perform gradient step
                loss = F.huber_loss(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            epoch += 1

            # Logging
            if epoch % 10_000 == 0 and epoch > 0:
                smoothed_reward = np.mean(rewards) if rewards else 0
                smoothed_rewards.append(smoothed_reward)
                total_rewards_list.append(rewards)
                rewards = []

                loss_mean = np.mean(losses) if losses else 0
                loss_means.append(loss_mean)
                total_loss_list.append(losses)
                losses = []

            if epoch % 100_000 == 0 and epoch > 0:
                # Save plots
                plt.plot(smoothed_rewards)
                plt.title("Average Reward on Breakout")
                plt.xlabel("Training Epochs [units of 10,000]")
                plt.ylabel("Average Reward per Episode")
                if not os.path.exists('./Imgs'):
                    os.makedirs('./Imgs')
                plt.savefig(f'./Imgs/average_reward_on_breakout_{epoch}.png')
                plt.close()

                plt.plot(loss_means)
                plt.title("Average Loss on Breakout")
                plt.xlabel("Training Epochs [units of 10,000]")
                plt.ylabel("Average Loss per Episode")
                plt.savefig(f'./Imgs/average_loss_on_breakout_{epoch}.png')
                plt.close()

                print(f"Epoch: {epoch}, Loss: {loss_mean}, Smoothed Reward: {smoothed_reward}")

            if epoch % 500_000 == 0 and epoch > 0:
                # Save checkpoint
                checkpoint_path = f'./checkpoints/dqn_checkpoint_{epoch}.pth'
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                if not os.path.exists('./checkpoint_data'):
                    os.makedirs('./checkpoint_data')
                torch.save(q_network.state_dict(), checkpoint_path)

                with open(f'./checkpoint_data/total_rewards_list_{epoch}.pkl', 'wb') as f:
                    pickle.dump(total_rewards_list, f)
                with open(f'./checkpoint_data/total_loss_list_{epoch}.pkl', 'wb') as f:
                    pickle.dump(total_loss_list, f)

            progress_bar.update(1)

        rewards.append(total_rewards)

        if total_rewards > best_reward:
            best_reward = total_rewards
            if not os.path.exists('./best_models'):
                os.makedirs('./best_models')
            torch.save(q_network.cpu(), f'./best_models/best_model_{best_reward}')
            q_network.to(device)


def evaluation(reward, visual=True, num_episode=10):
    # Initialize environment
    if visual:
        env_display = gym.make("ALE/Breakout-v5", render_mode="human")
    else:
        env_display = gym.make("ALE/Breakout-v5")
    env_display = gym.wrappers.RecordEpisodeStatistics(env_display)
    env_display = gym.wrappers.ResizeObservation(env_display, (84, 84))
    env_display = gym.wrappers.GrayscaleObservation(env_display)
    env_display = gym.wrappers.FrameStackObservation(env_display, 4)
    env_display = MaxAndSkipEnv(env_display, skip=4)

    obs, info = env_display.reset()
    rewards = []
    total_rewards = 0
    steps = 0

    # Load the trained model
    best_model_path = f"./best_models/best_model_{reward}"

    best_model = torch.load(best_model_path, map_location="cuda")
    best_model = best_model.to("cuda")
    best_model.eval()
    print(f"Episode Start")

    for episode in range(num_episode):
        print(episode)
        obs, info = env_display.reset()
        dead = False
        total_rewards = 0
        steps = 0

        while not dead:
            current_life = info['lives']
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cuda")
                q_values = best_model(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()  # Get action with max Q-value

            if steps == 0 or terminated or truncated:
                action = 1

            obs, reward, terminated, truncated, info = env_display.step(action)
            # print(action)
            dead = terminated or truncated
            total_rewards += reward
            steps += 1
            if visual:
                done = np.array(info['lives'] < current_life)

                if done:
                    print("life -1")

                if reward != 0:
                    print(f'Steps: {steps}, Reward: {reward}')

                if terminated or truncated:
                    print(f"Episode finished. Total rewards: {total_rewards}")

        rewards.append(total_rewards)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(mean_reward)
    # plt.plot(rewards)
    plt.plot(range(0, len(rewards)), rewards, label='Rewards per Episode')

    plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Mean: {mean_reward:.2f}, Std: {std_reward:.2f} ')

    plt.title("Best Model Reward on Breakout")
    plt.xlabel("Testing Epochs")
    plt.ylabel("Reward per Episode")
    if not os.path.exists('./Imgs'):
        os.makedirs('./Imgs')
    plt.legend()
    plt.show()
    plt.savefig(f'./Imgs/best_model_reward_on_breakout.png')
    plt.close()

    env_display.close()
    print(f"Evaluation completed.")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Current working directory:", os.getcwd())

    gym.register_envs(ale_py)

    env = gym.make("ALE/Breakout-v5")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)

    # Deep_Q_Learning(env, buffer_size=100_000, nb_epochs=5_000_000, replay_start_size=5_000, exploration_steps=1_000_000, device='cuda')

    best_rewards = 77.0
    evaluation(best_rewards, False, 1000)