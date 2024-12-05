# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import gymnasium as gym
import pytorch_lightning as pl

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN
from bc_model import BehavioralCloningModule

import torch
from torch.utils.data import Dataset, DataLoader

# %%
# Create the environment and PPO model
# Create and wrap the environment
env_name = "CarRacing-v3"
num_stack = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=4
)
vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")

# Create a PPO model with the same custom policy network
ppo_model = PPO.load("ppo_pt_car_racing", env=vec_env)
# %%

log_dir = Path("logs") / "fine_tuned" / f"{env_name}_stack{num_stack}"

# Update the environment and log settings for PPO fine-tuning
ppo_model.set_env(vec_env)
ppo_model.tensorboard_log = log_dir

# Fine-tune the model using PPO
ppo_model.learn(total_timesteps=500000)
ppo_model.save("ppo_fine_tuned_car_racing")
print(f"Model saved to ppo_fine_tuned_car_racing")

# %%
