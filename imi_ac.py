# %%
#!%load_ext autoreload
#!%autoreload 2

import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import gymnasium as gym
import pytorch_lightning as pl

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN

import torch
from torch.utils.data import Dataset, DataLoader


class RolloutDataset(Dataset):
    def __init__(self, s, a, r, s_prime, done, gamma=0.99):
        self.observations = torch.stack([x for x in s]).float()
        self.actions = torch.tensor(a).long()
        self.rewards = torch.tensor(r).float()
        self.next_observations = torch.stack([x for x in s_prime]).float()
        self.dones = torch.tensor(done).long()
        self.gamma = gamma

        self.discounted_returns = torch.zeros_like(self.rewards)
        self.ep_boundaries = torch.nonzero(self.dones).squeeze() + 1
        ep_start = 0
        for ei, ep_end in enumerate(self.ep_boundaries):
            rewards_episode = self.rewards[ep_start:ep_end]
            discounted_return = 0
            for i in reversed(range(len(rewards_episode))):
                discounted_return = rewards_episode[i] + (
                    self.gamma * discounted_return
                )
                self.discounted_returns[ep_start + i] = discounted_return
            ep_start = ep_end

    def visualize_discounted_returns(self):
        ep_start = 0
        for i, ep_end in enumerate(self.ep_boundaries):
            ep_dreturns = self.discounted_returns[ep_start:ep_end]
            plt.plot(ep_dreturns, label=f"Episode {i}")
            ep_start = ep_end
        plt.xlabel("Time Step")
        plt.ylabel("Discounted Return")
        plt.title("Discounted Returns for Each Episode")
        plt.show()

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.dones[idx],
            self.discounted_returns[idx],
        )


# Load the demonstrations
data = load_obj("processed_data/CarRacing-v3_processed.pkl")
s, a, r, s_prime, done = [], [], [], [], []
for i in range(len(data)):
    s.append(data[i][0])
    a.append(data[i][1])
    r.append(data[i][2])
    s_prime.append(data[i][3])
    done.append(data[i][4])

gamma = 0.99
dataset = RolloutDataset(s, a, r, s_prime, done, gamma=gamma)
# %%
dataset.visualize_discounted_returns()
# %%
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)


# Create and wrap the environment
env_name = "CarRacing-v3"
num_stack = 4
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, max_episode_steps=2000)
    ),
    n_envs=4,
)

vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")
# %%

# Load PPO model and extract the policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)


class PretrainACModule(pl.LightningModule):
    def __init__(self, net, lr=1e-3, gamma=0.99):
        super().__init__()
        self.net = net
        self.lr = lr
        self.gamma = gamma
        self.actor_loss_function = torch.nn.CrossEntropyLoss()
        self.critic_loss_function = torch.nn.MSELoss()

    def forward(self, x):
        # Return both the policy distribution logits and the value estimate
        actions, values, log_prob = self.net(x)
        dist = self.net.get_distribution(x)
        return dist.distribution.logits, values

    def training_step(self, batch, batch_idx=0):
        observations, actions, rewards, next_observations, dones, discounted_returns = (
            batch
        )

        # Get logits and value estimates
        logits, values = self(observations)

        # Actor loss (cross-entropy loss for discrete actions)
        actor_loss = self.actor_loss_function(logits, actions.long())

        # Critic loss (MSE between predicted and target values)
        target_values = discounted_returns.unsqueeze(1)
        critic_loss = self.critic_loss_function(values, target_values)

        # Total loss (actor + critic)
        total_loss = actor_loss + critic_loss

        # Log losses
        self.log(
            "train_actor_loss", actor_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_critic_loss",
            critic_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        return total_loss

    def configure_optimizers(self):
        # Use Adam optimizer for both actor and critic training
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# Create the BC Lightning Module
bc_module = PretrainACModule(
    net=model.policy,
    lr=1e-3,
    gamma=0.9,
)
# %%

pl_logger = pl.loggers.TensorBoardLogger("logs/imitation_learning_c", name=env_name)
trainer = pl.Trainer(max_epochs=100, logger=pl_logger)
trainer.fit(bc_module, data_loader)

# %%
bc_state_dict = {
    k[4:]: v for k, v in bc_module.net.state_dict().items() if k.startswith("net.")
}

model.policy.load_state_dict(bc_state_dict, strict=True)
model.save("ppo_pt_car_racing_ac")
# %%
