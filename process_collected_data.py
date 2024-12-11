# %%
#!%load_ext autoreload
#!%autoreload 2

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from utils import *

# %%
data_dir = Path("data_exp")
env_name = "CarRacing-v3"
consecutive_frames = 4
frame_step = 4
data_files = sorted(data_dir.glob(f"*car_racing_step{frame_step}_*.pkl"))
D = []
for data_file in data_files:
    data = load_obj(data_file)
    ep_reward = 0
    for i in range(len(data)):
        state, action, reward, next_state, done = data[i]
        state = torch.tensor(state, dtype=torch.float32)
        # all done states are at successful completions in this dataset
        if done:
            reward += 100
        ep_reward += reward
        D.append((state, action, reward, next_state, done))
    print(f"ep_reward: {ep_reward:.2f}")
print(f"Processed {len(D)} demonstrations")
# %%
processed_data_dir = Path("processed_data")
processed_data_dir.mkdir(exist_ok=True)
processed_data_file = processed_data_dir / f"{env_name}_{frame_step}_processed.pkl"
save_obj(D, processed_data_file)
print(f"Processed data saved to {processed_data_file}")
# %%
