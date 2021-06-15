from environment.config.base_config import BaseConfig
import numpy as np
from gym.spaces import Box

class BaseObservationConfig(BaseConfig):
    def __init__(self):
        # Gym Observation Space
        self.obs_space = None


class DefaultObservationConfig(BaseObservationConfig):
    def __init__(self):
        self.obs_dim = 3
        # speed steer delta_time
        self.obs_space = Box(low=np.tile(np.array([0.0, -0.5, -np.inf]), (self.frame_stack,1)),\
                             high=np.tile(np.array([1.0, 0.5, np.inf]), (self.frame_stack,1)), \
                             shape=(self.frame_stack, self.obs_dim), dtype=np.float32)

