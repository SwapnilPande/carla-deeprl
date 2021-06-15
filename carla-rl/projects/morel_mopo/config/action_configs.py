from environment.config.base_config import BaseConfig
import numpy as np
from gym.spaces import Box


class BaseActionConfig(BaseConfig):
    def __init__(self):

        # Gym Action Space
        self.action_space = None
        self.target_speed = None

class DefaultActionConfig(BaseActionConfig):
    def __init__(self):
        self.action_dim = 2 # TODO
        self.frame_stack = 2 #TODO
        self.action_space = Box(low=np.tile(np.array([-0.5, -0.5]), (self.frame_stack, 1)),\
                                high=np.tile(np.array([0.5, 0.5]), (self.frame_stack, 1)), \
                                shape=(self.frame_stack, self.action_dim), dtype=np.float32)
 
        self.target_speed = 20