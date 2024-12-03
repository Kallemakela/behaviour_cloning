import gymnasium as gym
import numpy as np
import torchvision.transforms as T


class TorchVisionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define the torchvision transformations (Resize and Grayscale)
        w = 48
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(num_output_channels=1),
                T.Resize((w, w), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
            ]
        )
        # Update observation space
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1, w, w), dtype=np.float32
        )

    def observation(self, obs):
        # Apply the transformation to the observation
        obs = self.transform(obs)
        return obs.numpy()
