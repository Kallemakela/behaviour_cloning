# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
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
eval_env = make_vec_env(
    lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=1
)
eval_env = VecFrameStack(eval_env, n_stack=num_stack, channels_order="first")
# %%
model_paths = {
    # "BC + Pretrained Critic": "ppo_pt_car_racing_ac",
    "Pure BC": "ppo_pt_car_racing",
    # "Fine-Tuned from BC": "ppo_fine_tuned_car_racing",
    # "Fine-Tuned from BC + Pretrained Critic": "ppo_fine_tuned_car_racing_ac",
    # "Baseline PPO": "CarRacing-v3",
}

# Initialize storage for results
evaluation_results = {}

# Load and evaluate each model
for model_name, model_path in model_paths.items():
    model = PPO.load(model_path, env=eval_env)
    # Evaluate the model using 20 episodes
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    evaluation_results[model_name] = (mean_reward, std_reward)
    print(f"{model_name}: Mean Reward: {mean_reward}, Std Reward: {std_reward}")

# Plotting the results for comparison
model_names = list(evaluation_results.keys())
mean_rewards = [evaluation_results[name][0] for name in model_names]
std_rewards = [evaluation_results[name][1] for name in model_names]

plt.figure(figsize=(10, 6))
plt.bar(model_names, mean_rewards, yerr=std_rewards, capsize=5, color="skyblue")
plt.xlabel("Models")
plt.ylabel("Average Reward")
plt.title("Model Comparison: Average Reward Across 20 Episodes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# %%
