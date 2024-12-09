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
frame_step = 1
D = []
for data_file in data_files:
    data = load_obj(data_file)
    ep_reward = 0
    for i in range(len(data)):
        frames = [
            i - j * frame_step for j in reversed(range(consecutive_frames))
        ]  # frames without padding
        state_frames = [max(0, frame) for frame in frames]  # pad front
        next_state_frames = [
            min(len(data) - 1, max(0, 1 + frame)) for frame in frames
        ]  # also pad end

        state = torch.stack([data[j][0] for j in state_frames]).float()
        # reshape from (consecutive_frames, 1, 64, 64) to (consecutive_frames*1, 64, 64)
        state = state.reshape(-1, *state.shape[2:])
        # padding at the end
        next_state = torch.stack(
            [data[min(1 + j, len(data) - 1)][0] for j in state_frames]
        ).float()
        next_state = next_state.reshape(-1, *next_state.shape[2:])

        action = data[i][1]
        reward = torch.tensor(data[i][2], dtype=torch.float32)
        done = data[i][4]
        # all done states are at successful completions in this dataset
        if done:
            reward += 100
        ep_reward += reward
        D.append((state, action, reward, next_state, done))
    print(f"ep_reward: {ep_reward:.2f}")
# %%
processed_data_dir = Path("processed_data")
processed_data_dir.mkdir(exist_ok=True)
processed_data_file = processed_data_dir / f"{env_name}_processed.pkl"
save_obj(D, processed_data_file)
print(f"Processed data saved to {processed_data_file}")
# %%
