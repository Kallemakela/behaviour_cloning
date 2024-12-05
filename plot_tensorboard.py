# %%
#!%load_ext autoreload
#!%autoreload 2

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

log_dirs = [
    Path("logs") / "baseline" / "CarRacing-v3_stack4" / "PPO_1",
    Path("logs") / "fine_tuned" / "CarRacing-v3_stack4" / "PPO_1",
    Path("logs") / "fine_tuned_ac" / "CarRacing-v3_stack4" / "PPO_1",
]
model_names = [
    "Baseline PPO",
    "FT from BC",
    "FT from BC + Pretrained Critic",
]
scalar_keys = [
    "rollout/ep_rew_mean",
]


def get_scalar_data(log_dir, scalar_key):
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()

    # Extract data for a specific scalar
    scalar_data = event_acc.Scalars(scalar_key)  # Example: first scalar
    steps = [item.step for item in scalar_data]
    values = [item.value for item in scalar_data]

    return steps, values


for log_dir, name in zip(log_dirs, model_names):
    steps, values = get_scalar_data(log_dir, scalar_keys[0])
    df = pd.DataFrame({"step": steps, "value": values})
    plt.plot(steps, values, label=name)

plt.xlabel("Steps")
plt.ylabel("Reward")
plt.title("CarRacing-v3: Episode mean reward")
plt.legend()
plt.savefig("car_racing_v3_episode_mean_reward.png")
plt.show()

# %%
