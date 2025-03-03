# %%
"""
Collects expert data using a trained model.
Uses a condition to save only good episodes. If a BC model is used this is basically iterative BC.
"""
#!%load_ext autoreload
#!%autoreload 2

from datetime import datetime
import time
import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
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
from utils import load_obj, save_obj
from baseline_model import CustomCNN
from custom_stack import VecFrameStepStack

# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# set_random_seed(seed)

env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4
n_ep = 200
max_steps = 2000
# render_mode = "human"
render_mode = "rgb_array"
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(
            env_name, continuous=False, render_mode=render_mode, max_episode_steps=1001
        )
    ),
    n_envs=1,
)
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)

save_path = f"ppo_pt_car_racing_step{frame_step}"
model = PPO.load(save_path, env=vec_env)

t0 = time.time()
for ei in range(n_ep):
    prev_obs = vec_env.reset()
    obs = prev_obs
    ep_r = 0
    ep_D = []
    for si in range(max_steps):
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = vec_env.step(action)
        ep_D.append((prev_obs[0], action[0], rewards[0], obs[0], dones[0]))
        prev_obs = obs
        ep_r += rewards
        # vec_env.render()
        if dones:
            break

    print(f"Episode {ei}: {ep_r}, Steps: {si + 1}")
    if si < 1000 and ep_r > 700:  # save only good episodes
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_dir = Path("data_exp")
        save_path = (
            save_dir / f"ppo_pt_car_racing_step{frame_step}_{timestamp}_{ei}.pkl"
        )
        save_obj(ep_D, save_path)

print(f"Time taken: {time.time()-t0:.2f}s")

vec_env.close()

# %%
