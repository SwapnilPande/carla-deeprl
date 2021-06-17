from environment.config.base_config import BaseConfig
import numpy as np
from gym.spaces import Box

class BaseObservationConfig(BaseConfig):
    def __init__(self):
        # Gym Observation Space
        self.obs_space = None


class DefaultObservationConfig(BaseObservationConfig):
    def __init__(self):
        self.obs_dim = 4
        self.frame_stack = 2

        # dist to traj, angle, speed, steer
        self.obs_space = Box(low=np.array([-np.inf,-np.inf,-np.inf,-np.inf]),\
                             high=np.array([np.inf, np.inf, np.inf, np.inf]), dtype=np.float32)
   