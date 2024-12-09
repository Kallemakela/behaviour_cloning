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

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN
from custom_stack import VecFrameStepStack

env_name = "CarRacing-v3"
num_stack = 4
frame_step = 1
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(
            env_name, continuous=False, render_mode="human", max_episode_steps=2000
        )
    ),
    n_envs=1,
)
# vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)
print("Action Space:", vec_env.action_space)
print("Observation Space:", vec_env.observation_space)
# check_env(vec_env)

save_path = "ppo_pt_car_racing_step1"
model = PPO.load(save_path, env=vec_env)

obs = vec_env.reset()
max_steps = 2000
for _ in range(max_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
    if dones:
        break

vec_env.close()
