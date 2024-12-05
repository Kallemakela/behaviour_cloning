# %%
"""
Custom env wrapper similar to VecFrameStep but skips n_step frames when stacking them.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations


class VecFrameStepStack(VecEnvWrapper):
    """
    Frame stacking wrapper for vectorized environment. Designed for image observations.

    :param venv: Vectorized environment to wrap
    :param n_stack: Number of frames to stack
    :param n_step: Number of steps to skip when stacking frames
    :param channels_order: If "first", stack on first image dimension. If "last", stack on last dimension.
        If None, automatically detect channel to stack over in case of image observation or default to "last" (default).
        Alternatively channels_order can be a dictionary which can be used with environments with Dict observation spaces
    """

    def __init__(
        self,
        venv: VecEnv,
        n_stack: int,
        channels_order: Optional[Union[str, Mapping[str, str]]] = None,
        n_step: int = 4,
    ) -> None:
        assert isinstance(
            venv.observation_space, (spaces.Box, spaces.Dict)
        ), "VecFrameStack only works with gym.spaces.Box and gym.spaces.Dict observation spaces"

        self.stacked_obs = StackedObservations(
            venv.num_envs, n_stack * n_step, venv.observation_space, channels_order
        )
        low = np.repeat(venv.observation_space.low, n_stack, axis=0)
        high = np.repeat(venv.observation_space.high, n_stack, axis=0)
        stacked_observation_space = spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        self.n_step = n_step
        super().__init__(venv, observation_space=stacked_observation_space)

    def step_wait(
        self,
    ) -> Tuple[
        Union[np.ndarray, Dict[str, np.ndarray]],
        np.ndarray,
        np.ndarray,
        List[Dict[str, Any]],
    ]:
        observations, rewards, dones, infos = self.venv.step_wait()
        observations, infos = self.stacked_obs.update(observations, dones, infos)  # type: ignore[arg-type]
        # pick last observation and every n_step-th observation
        observations = observations[:, self.n_step - 1 :: self.n_step]
        return observations, rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Reset all environments
        """
        observation = self.venv.reset()
        observation = self.stacked_obs.reset(observation)  # type: ignore[arg-type]
        observation = observation[:, self.n_step - 1 :: self.n_step]
        return observation
