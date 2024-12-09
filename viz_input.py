# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import load_obj
import matplotlib.patches as patches

# %%
data = load_obj("processed_data/CarRacing-v3_4_processed_0.pkl")
# %%
obs = data[110][0]
n_frames, h, w = obs.shape

fig, ax = plt.subplots(1, n_frames, figsize=(12, 4))
for i in range(n_frames):
    ax[i].imshow(obs[i].squeeze(0).numpy(), cmap="gray")
    ax[i].axis("off")
plt.tight_layout()
# plt.savefig("car_racing_v3_obs.png")
plt.show()

# %%
