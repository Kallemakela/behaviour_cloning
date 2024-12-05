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
from model import QNetworkCNN

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


# %%
for i, batch in enumerate(dataloader):
    break
model = QNetworkCNN.load_from_checkpoint(
    "logs/q_network_cnn/version_1/checkpoints/epoch=9-step=640.ckpt"
)
# %%
states, actions, rewards, next_states, dones = batch
with torch.inference_mode():
    q_values = model(states)
    next_q_values = model(next_states)
# %%
q_values
rewards
