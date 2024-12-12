# %%
#!%load_ext autoreload
#!%autoreload 2


from stable_baselines3.common.evaluation import evaluate_policy
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
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


class LazyExpertDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # observation = torch.tensor(item[0], dtype=torch.float32)
        observation = item[0]
        # action = torch.tensor(item[1], dtype=torch.long)
        action = item[1]
        return observation, action


# %%

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")
# %%
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
set_random_seed(seed)

env_name = "CarRacing-v3"
num_stack = 4
frame_step = 4

# Load the demonstrations
# data = load_obj("processed_data/CarRacing-v3_processed.pkl")
data_paths = [
    # f"/scratch/work/makelak6/datasets/RL/CarRacing-v3_{frame_step}_processed_0.pkl"
    f"processed_data/CarRacing-v3_4_processed.pkl"
]
data = []
for dpath in data_paths:
    data.extend(load_obj(dpath))
print(f"Loaded {len(data)} demonstrations")
dataset = LazyExpertDataset(data)
print(f"Dataset created with {len(dataset)} samples")
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
print(f"Data loader created with {len(data_loader)} batches")

vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, max_episode_steps=2000)
    ),
    n_envs=1,
)
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)
print(f"Observation space: {vec_env.observation_space.shape}")
# %%
# Load PPO model and extract the policy
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)
model = PPO("CnnPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)
policy_network = model.policy

# from pytorch_lightning.tuner.tuning import Tuner
# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(bc_module, data_loader, num_training=1000)
# lr = lr_finder.suggestion()
# print(f"Learning rate suggestion: {lr}")

bc_module = BehavioralCloningModule(policy_network, lr=2e-4)

# %%
pl_logger = pl.loggers.TensorBoardLogger("logs/imitation_learning", name=env_name)
trainer = pl.Trainer(
    max_epochs=200,
    accelerator="gpu",
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
