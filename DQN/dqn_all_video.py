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
        self.network = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
                                     nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
                                     nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
                                     nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(),
                                     nn.Linear(512, nb_actions),)

    def forward(self, x):
        return self.network(x / 255.)

def evaluation(reward):


    # Initialize environment
    env_display = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env_display = gym.wrappers.RecordVideo(
        env_display,
        episode_trigger=lambda num: num % 2 == 0,
        video_folder="saved-video-folder",
        name_prefix="struggle_video-",
    )
    env_display = gym.wrappers.RecordEpisodeStatistics(env_display)
    env_display = gym.wrappers.ResizeObservation(env_display, (84, 84))
    env_display = gym.wrappers.GrayscaleObservation(env_display)
    env_display = gym.wrappers.FrameStackObservation(env_display, 4)
    env_display = MaxAndSkipEnv(env_display, skip=4)

    obs, info = env_display.reset()

    total_rewards = 0
    steps = 0

    # Load the trained model
    best_model_path = f"./best_models/best_model_{reward}"

    best_model = torch.load(best_model_path, map_location="cuda")
    best_model = best_model.to("cuda")
    best_model.eval()
    print(f"Episode Start")
    for episode in range(10):
        obs, info = env_display.reset()
        dead = False
        while not dead:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to("cuda")
                q_values = best_model(obs_tensor)
                action = torch.argmax(q_values, dim=1).item()  # Get action with max Q-value

            if steps == 0 or terminated or truncated:
                action = 1

            obs, reward, terminated, truncated, info = env_display.step(action)
            dead = terminated or truncated
            total_rewards += reward
            steps += 1

            if reward != 0:
                print(f'Steps: {steps}, Reward: {reward}')

            if terminated or truncated:
                print(f"Episode finished. Total rewards: {total_rewards}")
                total_rewards = 0
                steps = 0

    env_display.close()



if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Breakout Game Initialisation
    gym.register_envs(ale_py)
    env = gym.make("ALE/Breakout-v5")

    # # Preprocessing

    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)


    # # Evaluation
    best_rewards = 10.0
    evaluation(best_rewards)


