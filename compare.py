# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import seaborn as sns
import gymnasium as gym
import pytorch_lightning as pl

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN
from bc_model import BehavioralCloningModule

import torch
from torch.utils.data import Dataset, DataLoader
from custom_stack import VecFrameStepStack

np.random.seed(0)
torch.manual_seed(0)

# %%

env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4

model_paths = {
    "Pure BC": "ppo_pt_car_racing_step4",
    # "256": "ppo_pt_car_racing_step4_256",
    # "Pure BC (Step 4)": "ppo_pt_car_racing_step4",
    # "Pure BC (Step 10)": "ppo_pt_car_racing_step10",
    # "CAC": "ppo_pt_car_racing_ac",
    # "FT BC": "ppo_fine_tuned_car_racing",
    # "FT BC": "ppo_fine_tuned_car_racing_step4",
    # "FT CAC": "ppo_fine_tuned_car_racing_ac",
    # "Base": "CarRacing-v3",
}

eval_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, max_episode_steps=2000)
    ),
    n_envs=1,
)
# eval_env = VecFrameStack(eval_env, n_stack=num_stack, channels_order="first")
eval_env = VecFrameStepStack(
    eval_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)

# Initialize storage for results
evaluation_results = {}

# Load and evaluate each model
n_eval_episodes = 50
for mi, (model_name, model_path) in enumerate(model_paths.items()):

    model = PPO.load(model_path, env=eval_env)
    # mean_reward, std_reward, rewards = evaluate_policy(
    rewards, ep_lens = evaluate_policy(
        model, eval_env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    evaluation_results[model_name] = rewards
    mu = np.mean(rewards)
    sigma = np.std(rewards)
    ste = sigma / np.sqrt(n_eval_episodes)
    median = np.median(rewards)
    print(f"{model_name}: Mean Reward: {mu:.2f} +/- {ste:.2f}, med: {median:.2f}")

# %%
# Plotting the results for comparison
model_names = list(evaluation_results.keys())
plot_df = pd.DataFrame(
    {
        "Model": np.repeat(model_names, n_eval_episodes),
        "Episode": np.tile(range(1, n_eval_episodes + 1), len(model_names)),
        "Reward": np.concatenate(
            [evaluation_results[model_name] for model_name in model_names]
        ),
    }
)
sns.swarmplot(data=plot_df, x="Model", y="Reward", alpha=0.5, color="C0")
sns.violinplot(data=plot_df, x="Model", y="Reward", color="C1")
plt.title("CarRacing-v3: Episode Rewards")
plt.savefig(f"car_racing_v3_episode_rewards_step.png")
# plt.show()
plt.close()

# %%
