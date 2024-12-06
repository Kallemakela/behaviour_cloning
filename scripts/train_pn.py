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
from scripts.model import PolicyNetworkCNN

# %%
env_name = "Skiing-v5"
data_file = Path(f"processed_data/{env_name}_processed.pkl")
data = load_obj(data_file)

# %%
episode_ends = [i for i, d in enumerate(data) if d[4]]
d = data[episode_ends[0]]
state, action, reward, next_state, done = d
print(state.shape, action, reward, next_state.shape, done)

# fig, ax = plt.subplots(1, 4, figsize=(12, 3))
# for frame_i in range(4):
#     ax[frame_i].imshow(d[0][frame_i].squeeze(0).numpy(), cmap="gray")
#     ax[frame_i].axis("off")
# plt.show()

dataset = RLDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


for i, batch in enumerate(dataloader):
    break
model = PolicyNetworkCNN(input_shape=(4, 64, 64), num_actions=3)
model.training_step(batch)
model.validation_step(batch)
# %%
pl.seed_everything(1)
pl_logger = pl.loggers.CSVLogger("logs", name=f"policy_network_cnn_{env_name}")
trainer = pl.Trainer(
    max_epochs=10,
    # accelerator="cpu",
    logger=pl_logger,
)
trainer.fit(model, dataloader, dataloader)


# %%
