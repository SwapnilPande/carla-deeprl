import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import scipy.spatial

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gym
from gym.spaces import Box, Discrete, Tuple

# compute reward
from projects.morel_mopo.algorithm.reward import compute_reward

# Import BaseFakeEnv and utils
from projects.morel_mopo.algorithm.fake_envs.base_fake_env import BaseFakeEnv


from projects.morel_mopo.config.data_module_config import MixedProbabilisticGRUDataModuleConfig
from projects.morel_mopo.algorithm.data_modules import RNNOfflineCarlaDataModule
from projects.morel_mopo.algorithm.fake_envs.fake_env_utils import filter_waypoints



class FakeEnv(BaseFakeEnv):
    def __init__(self, dynamics,
                        config,
                        logger = None):

        # This is the order in which new observations/actions are added to the history
        # New first means the newest observation/action is at the front of the list
        self.NEW_FIRST = True
        super(FakeEnv, self).__init__(dynamics, config, logger)

    '''
    Updates state vector according to dynamics prediction
    # @params: delta      [Δspeed, Δsteer]
    # @return new state:  [[speed_t+1, steer_t+1], [speed_t, steer_t], speed_t-1, steer_t-1]]
    '''
    def update_next_state(self, prev_state, delta_state):
        # import ipdb; ipdb.set_trace()
        # calculate newest state
        newest_state = prev_state.unnormalized[..., 0, :] + delta_state

        # insert newest state at front
        return torch.cat([newest_state.unsqueeze(-2), prev_state.unnormalized[..., :-1, :]], dim=-2)

    def update_action(self, prev_action, new_action):
        # insert new action at front, delete oldest action
        try:
            return torch.cat(
                        [new_action.unsqueeze(-2),
                        prev_action.unnormalized[..., :-1, :]],
                        dim = -2
                )
        except:
            import ipdb; ipdb.set_trace()

    def make_prediction(self, past_state, past_action):
        # Get predictions across all models
        try:
            states, rewards = self.dynamics.predict(past_state.normalized, past_action.normalized)
            states = torch.stack(states)
            return states
        except:
            import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    dm_config = MixedProbabilisticGRUDataModuleConfig()
    dm_config.batch_size = 1

    dm = RNNOfflineCarlaDataModule(dm_config)

    dm.setup()


    import numpy as np
    import matplotlib.pyplot as plt


    for i in range(50):
        print(i)
        ((obs, action, _, _, _, vehicle_pose, mask), waypoints) = dm.sample_with_waypoints()
        import ipdb; ipdb.set_trace()

        waypoints = filter_waypoints(waypoints)

        x = waypoints[:,0]
        y = waypoints[:,1]

        plt.figure()
        plt.scatter(x,y)
        plt.savefig("/home/scratch/swapnilp/test.png")
        plt.close()

        input()

