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
from env import TorchVisionWrapper
from custom_stack import VecFrameStepStack

# %%
env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=1
)
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)

print("Action Space:", vec_env.action_space)
print("Observation Space:", vec_env.observation_space)

# %% Train the agent
log_dir = Path("logs") / "baseline" / f"{env_name}_stack{num_stack}_step{frame_step}"

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO(
    "CnnPolicy",
    vec_env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=policy_kwargs,
)
model.learn(total_timesteps=500000)
save_path = f"ppo_baseline_step{frame_step}"
model.save(save_path)
print(f"Model saved to {save_path}")
# %%
