# %%
"""
Script for testing custom env
"""
#!%load_ext autoreload
#!%autoreload 2


import numpy as np
from gymnasium import spaces
import gymnasium as gym

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from custom_stack import VecFrameStepStack


# %%
class StepIndexEnv(gym.Env):
    def __init__(self):
        super(StepIndexEnv, self).__init__()
        # Define action and observation space
        # Here we'll assume a simple discrete action space with 2 actions (e.g., 0 or 1)
        self.action_space = spaces.Discrete(2)
        # Observation space is just a single integer (the step index)
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int32
        )
        self.current_step = 0

    def reset(self, **kwargs):
        # Reset the state of the environment to the initial state
        self.current_step = 0
        return np.array([self.current_step], dtype=np.int32), {}

    def step(self, action):
        # Increment the step index
        self.current_step += 1
        obs = np.array([self.current_step], dtype=np.int32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Simple rendering to console
        print(f"Current Step: {self.current_step}")

    def close(self):
        pass


n_stack = 4
n_step = 4
n_iters = 20
vec_env = make_vec_env(lambda: StepIndexEnv(), n_envs=1)

# env_name = "CarRacing-v3"
# from env import TorchVisionWrapper
# vec_env = make_vec_env(
#     lambda: TorchVisionWrapper(gym.make(env_name, continuous=False)), n_envs=1
# )

vec_env = VecFrameStepStack(vec_env, n_stack=n_stack, n_step=4, channels_order="first")
obs = vec_env.reset()

for _ in range(n_iters):
    action = np.array([0])
    obs, reward, done, info = vec_env.step(action)
    print(obs)
    if done:
        obs = vec_env.reset()
# %%
