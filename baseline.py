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

from baseline_model import CustomCNN


# %%

# env_name = "CarRacing-v3"
# env = gym.make(env_name)
# wrapped_env = TorchVisionWrapper(env)

# obs = wrapped_env.reset()[0]  # Reset and get the initial observation
# for _ in range(5):
#     # Take random actions to get a sequence of frames
#     action = wrapped_env.action_space.sample()
#     obs, reward, done, truncated, info = wrapped_env.step(action)

#     # Convert observation back to a numpy image to plot using matplotlib
#     img = np.transpose(
#         obs, (1, 2, 0)
#     )  # Convert shape from (1, 84, 84) to (84, 84, 1) for plotting

#     plt.imshow(img.squeeze(), cmap="gray")
#     plt.title("Transformed Observation (Grayscale & Resized)")
#     plt.axis("off")
#     plt.show()

#     if done:
#         obs = wrapped_env.reset()[0]

# obs.shape, obs.min(), obs.max()
# %%
env_name = "CarRacing-v3"
num_stack = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=4
)

vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")

print("Action Space:", vec_env.action_space)
print("Observation Space:", vec_env.observation_space)

# %% Train the agent
log_dir = Path("logs") / "baseline" / f"{env_name}_stack{num_stack}"

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
# model = PPO(
#     "CnnPolicy",
#     vec_env,
#     verbose=1,
#     tensorboard_log=log_dir,
#     policy_kwargs=policy_kwargs,
# )
model = PPO.load(env_name, env=vec_env)
model.learn(total_timesteps=200000)
model.save(env_name)
print(f"Model saved to {env_name}")
# %%
