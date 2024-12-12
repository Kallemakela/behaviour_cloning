# %%
from stable_baselines3 import PPO
from pathlib import Path

# %%
model_paths = {
    "Pure BC": "ppo_pt_car_racing_step4",
    "256": "ppo_pt_car_racing_step4_256",
    "big": "ppo_pt_car_racing_step4_big",
    "small": "ppo_pt_car_racing_step4_small",
    # "Pure BC (Step 4)": "ppo_pt_car_racing_step4",
    # "Pure BC (Step 10)": "ppo_pt_car_racing_step10",
    # "CAC": "ppo_pt_car_racing_ac",
    # "FT BC": "ppo_fine_tuned_car_racing",
    # "FT BC": "ppo_fine_tuned_car_racing_step4",
    # "FT CAC": "ppo_fine_tuned_car_racing_ac",
    # "Base": "CarRacing-v3",
}
model_path = model_paths["big"]
# model_path = model_paths["small"]
model = PPO.load(model_path)
total_params = 0
for param, value in model.policy.named_parameters():
    if "features_extractor" in param:
        print(param, value.shape)
        total_params += value.numel()
total_params
# %%
