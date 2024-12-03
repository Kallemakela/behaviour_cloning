# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import *

# %%
data_dir = Path("data")
env_name = "CarRacing-v3"
data_files = sorted(data_dir.glob(f"{env_name}_*.pkl"))
consecutive_frames = 4
D = []
for data_file in data_files:
    data = load_obj(data_file)
    for i in range(len(data) - consecutive_frames):
        state = torch.stack([data[i + j][0] for j in range(consecutive_frames)]).float()
        # reshape from (consecutive_frames, 1, 64, 64) to (consecutive_frames*1, 64, 64)
        state = state.reshape(-1, *state.shape[2:])
        action = data[i + consecutive_frames][1]
        reward = torch.tensor(data[i + consecutive_frames][2], dtype=torch.float32)
        next_state = torch.stack(
            [data[i + j][0] for j in range(1, consecutive_frames + 1)]
        ).float()
        next_state = next_state.reshape(-1, *next_state.shape[2:])
        done = data[i + consecutive_frames][4]
        D.append((state, action, reward, next_state, done))
    print(f"{data_file.name} processed")
# %%
processed_data_dir = Path("processed_data")
processed_data_dir.mkdir(exist_ok=True)
processed_data_file = processed_data_dir / f"{env_name}_processed.pkl"
save_obj(D, processed_data_file)
print(f"Processed data saved to {processed_data_file}")
# %%
