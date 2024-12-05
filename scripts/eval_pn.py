# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl

from utils import *
from dataset import RLDataset
from model import PolicyNetworkCNN

# %%
data_file = Path("processed_data/Freeway-v5_20241130172648_processed.pkl")
data = load_obj(data_file)
dataset = RLDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
# %%
d = data[0]
state, action, reward, next_state, done = d
print(state.shape, action, reward, next_state.shape, done)
# %%
data[0][0].shape
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for frame_i in range(4):
    ax[frame_i].imshow(data[20][0][frame_i].squeeze(0).numpy(), cmap="gray")
    ax[frame_i].axis("off")
plt.show()

model = PolicyNetworkCNN.load_from_checkpoint(
    "logs/policy_network_cnn/version_3/checkpoints/epoch=5-step=384.ckpt"
)
# %%
for i, batch in enumerate(dataloader):
    states, actions, rewards, next_states, dones = batch
    with torch.inference_mode():
        action_probs = torch.softmax(model(states), dim=-1)
    pred_action = torch.argmax(action_probs, dim=-1)
    if not torch.all(pred_action == 1):
        break

# %%
