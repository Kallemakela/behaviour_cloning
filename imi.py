# %%
#!%load_ext autoreload
#!%autoreload 2


from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import gymnasium as gym
import pytorch_lightning as pl

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN, CustomCNN2
from bc_model import BehavioralCloningModule
from custom_stack import VecFrameStepStack

import torch
from torch.utils.data import Dataset, DataLoader


class EvalCallback(pl.Callback):
    def __init__(self, sb_model, env, interval=50):
        self.sb_model = sb_model
        self.env = env
        self.interval = interval

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.interval == 0:
            pl_module_state_dict = {
                k.replace("policy_network.", ""): v
                for k, v in pl_module.state_dict().items()
                if k.startswith("policy_network")
            }
            self.sb_model.policy.load_state_dict(pl_module_state_dict, strict=True)
            rewards, ep_lens = evaluate_policy(
                self.sb_model, self.env, n_eval_episodes=10, return_episode_rewards=True
            )
            self.log("eval_reward", np.mean(rewards), on_step=False, on_epoch=True)
            self.log("eval_ep_len", np.mean(ep_lens), on_step=False, on_epoch=True)
            self.log(
                "eval_median_reward", np.median(rewards), on_step=False, on_epoch=True
            )


class ExpertDataset(Dataset):
    def __init__(self, observations, actions):
        self.observations = observations
        self.actions = actions

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4

# Load the demonstrations
# data = load_obj("processed_data/CarRacing-v3_processed.pkl")
data_paths = [f"processed_data/CarRacing-v3_{frame_step}_processed_0.pkl"]
data = []
for dpath in data_paths:
    data.extend(load_obj(dpath))

# data = data[10000:40000]  # works
# data = data[:10000]  # works
data = data[:100000]  # does not work
observations = torch.stack([d[0] for d in data]).float()
actions = torch.tensor([d[1] for d in data]).long()
# %%
mu, std = observations.mean(), observations.std()
observations = (observations - mu) / std
# %%
# Create the dataset and data loader
dataset = ExpertDataset(observations, actions)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, max_episode_steps=2000)
    ),
    n_envs=1,
)
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)
# %%
# Load PPO model and extract the policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN2,
    features_extractor_kwargs=dict(features_dim=512),
)
model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)

# Extract the policy network
policy_network = model.policy

# Create the BC Lightning Module
bc_module = BehavioralCloningModule(policy_network)
# %%
# for i, (obs, act) in enumerate(data_loader):
#     break
# bc_module.training_step((obs, act), i)

# %%
pl_logger = pl.loggers.TensorBoardLogger("logs/imitation_learning", name=env_name)
trainer = pl.Trainer(
    max_epochs=100,
    logger=pl_logger,
    # callbacks=[EvalCallback(model, vec_env)]
)
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
