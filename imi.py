# %%
#!%load_ext autoreload
#!%autoreload 2

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
from custom_stack import VecFrameStepStack

import torch
from torch.utils.data import Dataset, DataLoader


class ExpertDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


# Load the demonstrations
data = load_obj("processed_data/CarRacing-v3_processed.pkl")
observations = torch.stack([d[0] for d in data])
actions = torch.tensor([d[1] for d in data])
# %%
# Create the dataset and data loader
dataset = ExpertDataset(observations, actions)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=1
)
# eval_env = VecFrameStack(eval_env, n_stack=num_stack, channels_order="first")
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)
# %%

# Load PPO model and extract the policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

# Extract the policy network
policy_network = model.policy

# Create the BC Lightning Module
bc_module = BehavioralCloningModule(policy_network)

pl_logger = pl.loggers.TensorBoardLogger("logs/imitation_learning", name=env_name)
trainer = pl.Trainer(max_epochs=100, logger=pl_logger)
trainer.fit(bc_module, data_loader)
trainer.save_checkpoint("bc_model.ckpt")

# %%
bc_state_dict = {
    k.replace("policy_network.", ""): v for k, v in bc_module.state_dict().items()
}
model.policy.load_state_dict(bc_state_dict, strict=True)
save_path = f"ppo_pt_car_racing_step{frame_step}"
model.save(save_path)
print(f"Model saved at {save_path}")
# %%
