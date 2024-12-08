# %%
#!%load_ext autoreload
#!%autoreload 2

from scipy.ndimage import zoom
import cv2
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pathlib import Path
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_checker import check_env
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
from pathlib import Path

from env import TorchVisionWrapper
from utils import load_obj
from baseline_model import CustomCNN
from custom_stack import VecFrameStepStack

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
set_random_seed(seed)


# %%
def get_cam(model, x, layer):
    features = []
    gradients = []

    # Hooks to extract the features and gradients
    def forward_hook(module, input, output):
        features.append(output)

    def register_full_backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    handle_forward = layer.register_forward_hook(forward_hook)
    handle_backward = layer.register_backward_hook(register_full_backward_hook)

    # Forward pass
    dist = model.get_distribution(x)
    logits = dist.distribution.logits
    a = logits.argmax(dim=1)
    target = logits[0, a]

    # Backward pass (gradient calculation for the target class)
    model.zero_grad()
    target.backward(retain_graph=True)

    # Extract the features and gradients
    feature_map = features[0].squeeze(0)
    gradient_map = gradients[0].squeeze(0)
    # print(gradient_map.shape, gradient_map.min(), gradient_map.max())

    # Global Average Pooling over gradients
    weights = torch.mean(gradient_map, dim=(1, 2))

    # Compute the CAM
    cam = torch.zeros(feature_map.shape[1:], dtype=feature_map.dtype)
    for i, w in enumerate(weights):
        cam += w * feature_map[i]

    # Apply ReLU to the CAM (since we want only positive influences)
    cam = F.relu(cam)

    # Normalize the CAM to the range [0, 1]
    cam -= cam.min()
    cam /= cam.max() + 1e-5

    # Resize the CAM to match the original input size using scipy zoom
    cam = cam.detach().cpu().numpy()
    input_size = x.shape[2:]  # Assuming x has shape [batch, channels, height, width]
    # print(cam.shape, input_size)
    cam_resized = zoom(
        cam, (input_size[0] / cam.shape[0], input_size[1] / cam.shape[1])
    )

    # Remove hooks to avoid any potential memory issues
    handle_forward.remove()
    handle_backward.remove()

    return cam_resized


def visualize_cam(cam, observation):
    if observation.shape[0] == 4:
        observation = observation[-1]
    # plt.figure(figsize=(10, 5))
    # plt.imshow(observation, alpha=0.8)
    # plt.imshow(cam, cmap="jet", alpha=0.4)
    # plt.title("Class Activation Map Overlay")
    # plt.axis("off")
    # plt.show()
    fig, ax = plt.subplots()
    ax.imshow(observation)
    ax.imshow(cam, cmap="jet", alpha=0.4)
    ax.axis("off")
    fig.tight_layout()
    return fig


env_name = "CarRacing-v3"
num_stack = 4
frame_step = 1
vec_env = make_vec_env(
    lambda: TorchVisionWrapper(
        gym.make(env_name, continuous=False, render_mode="rgb_array")
    ),
    n_envs=1,
)
# vec_env = VecFrameStack(vec_env, n_stack=num_stack, channels_order="first")
vec_env = VecFrameStepStack(
    vec_env, n_stack=num_stack, n_step=frame_step, channels_order="first"
)
print("Action Space:", vec_env.action_space)
print("Observation Space:", vec_env.observation_space)


# save_path = "ppo_pt_car_racing_step1"
# save_path = "ppo_baseline"
save_path = "ppo_fine_tuned_car_racing"
fig_path = Path("fig/cams") / save_path
fig_path.mkdir(parents=True, exist_ok=True)
model = PPO.load(save_path, env=vec_env)

obs = vec_env.reset()
max_steps = 2000
for _ in range(max_steps):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    if _ % 1 == 0:
        # CNN layers:
        # model.policy.pi_features_extractor.cnn[2], out size [11, 11]
        # model.policy.pi_features_extractor.cnn[2], out size [4, 4]
        # model.policy.pi_features_extractor.cnn[4], out size [2, 2]
        l = model.policy.pi_features_extractor.cnn[0]
        o = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
        cam = get_cam(model.policy, o, l)
        orig_img = vec_env.render()
        w, h, c = orig_img.shape
        cam_resized = zoom(cam, (w / cam.shape[0], h / cam.shape[1]))
        # fig = visualize_cam(cam, obs.squeeze(0))
        fig = visualize_cam(cam_resized, orig_img)
        fig.savefig(fig_path / f"cam_{_}.png")
        # plt.show()
        plt.close(fig)

    if dones:
        break


vec_env.close()
# %%
