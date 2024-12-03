# %%
#!%load_ext autoreload
#!%autoreload 2

from torchvision import transforms
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from pathlib import Path
import pygame

import ale_py

gym.register_envs(ale_py)

from utils import *

pygame.init()


env_name = "CarRacing-v3"
env = gym.make(f"{env_name}", render_mode="human", continuous=False)


transform = get_img_transform()

# plt.imshow(transform(state).squeeze(0).numpy(), cmap="gray")

# %%
done = False
action = 0  # Default action

D = []
max_steps = int(1e8)
state, info = env.reset()
prev_state = state
total_reward, avg_reward = 0, 0
for step_i in range(max_steps):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    keys = pygame.key.get_pressed()
    if keys[pygame.K_RIGHT]:
        action = 1
    elif keys[pygame.K_LEFT]:
        action = 2
    elif keys[pygame.K_UP]:
        action = 3
    elif keys[pygame.K_z]:
        action = 4
    elif keys[pygame.K_ESCAPE]:
        break
    else:
        action = 0  # Default NOOP action

    # Step the environment
    state, reward, done, truncated, info = env.step(action)
    D.append((transform(prev_state), action, reward, transform(state), done))

    avg_reward += reward
    total_reward += reward
    if step_i % 100 == 0:
        print(f"Step: {step_i}, Avg Reward: {avg_reward / 100}")
        avg_reward = 0

    # if step_i % 1 == 0:
    #     processed_state = transform(state)
    #     # processed_prev_state = transform(prev_state)
    #     plt.imshow(processed_state.squeeze(0).numpy(), cmap="gray", vmin=0, vmax=1)
    #     # diff = transform(state - prev_state)
    #     # plt.imshow(diff.squeeze(0).numpy(), cmap="gray")
    #     plt.savefig(f"fig/step_{step_i}.png")
    #     plt.close()

    prev_state = state
    if done:
        break

print(f"Step: {step_i}, Total Reward: {avg_reward}")

# Clean up
env.close()
pygame.quit()

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
save_obj(D, str(data_dir / f"{env_name}_{timestamp}.pkl"))
# %%
