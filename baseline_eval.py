# %%
#!%load_ext autoreload
#!%autoreload 2

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# %%
class TorchVisionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define the torchvision transformations (Resize and Grayscale)
        w = 48
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(num_output_channels=1),
                T.Resize((w, w), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, w, w), dtype=np.float32
        )

    def observation(self, obs):
        # Apply the transformation to the observation
        obs = self.transform(obs)
        return obs.numpy()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate the output size after the CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Fully connected layer to map to features_dim
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))


env_name = "CarRacing-v3"
num_stack = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, render_mode="human")
    ),
    n_envs=1,
)
vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")

print("Action Space:", vec_env.action_space)
print("Observation Space:", vec_env.observation_space)
# check_env(vec_env)

model = PPO.load(env_name, env=vec_env)
# %%
obs = vec_env.reset()
max_steps = 2000
for _ in range(max_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
