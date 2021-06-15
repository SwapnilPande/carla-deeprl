import os
import sys
import glob


# Import other necessary configs
from environment.config.base_config import BaseConfig
from environment.config import observation_configs, action_configs, reward_configs


class BaseFakeEnvConfig(BaseConfig):
    """Base Class defining the parameters required in main config.

    DO NOT instantiate this directly. Instead, using DefaultMainConfig
    """
    def __init__(self):
        self.reward_config = None
        self.obs_config = None
        self.action_config = None
        self.uncertainty_config = None

        self.uncertainty_coeff = None
        self.timeout_steps = None


    def populate_config(self, observation_config = 'LowDimObservationConfig', action_config = 'MergedSpeedTanhConfig', reward_config = 'Simple2RewardConfig'):
        """Fill in the config parameters that are not set by default

        For each type of config, the parameter can be either passed in as a string containing the class name or
        an instance of the config type. The config must be a subclass of the respective config base class.
        Ex: observation_config can be a string "LowDimObsConfig" or it can be a class (or subclass) of type BaseObsConfig
        """
        # Observation Config
        if(isinstance(observation_config, str)):
            # Get reference to object
            config_type = getattr(observation_configs, observation_config)

            # Instantiate Object
            self.obs_config = config_type()
        elif(isinstance(observation_config, observation_configs.BaseObservationConfig)):
            # Just save object, since it is already instantiated
            self.obs_config = observation_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for observation_config")

        # Action Config
        if(isinstance(action_config, str)):
            # Get reference to object
            config_type = getattr(action_configs, action_config)

            # Instantiate Object
            self.action_config = config_type()
        elif(isinstance(action_config, action_configs.BaseActionConfig)):
            # Just save object, since it is already instantiated
            self.action_config = action_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for action_config")

        # Reward Config
        if(isinstance(reward_config, str)):
            # Get reference to object
            config_type = getattr(reward_configs, reward_config)

            # Instantiate Object
            self.reward_config = config_type()
        elif(isinstance(reward_config, reward_configs.BaseRewardConfig)):
            # Just save object, since it is already instantiated
            self.reward_config = reward_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for reward_config")

        # Uncertainty Config
        if(isinstance(uncertainty_config, str)):
            # Get reference to object
            config_type = getattr(uncertainty_configs, uncertainty_config)

            # Instantiate Object
            self.uncertainty_config = config_type()
        elif(isinstance(uncertainty_config, uncertainty_configs.BaseUncertaintyConfig)):
            # Just save object, since it is already instantiated
            self.uncertainty_config = uncertainty_config
        else:
            # Invalid Argument
            raise Exception("Invalid argument for uncertainty_config")


class DefaultFakeEnvConfig(BaseFakeEnvConfig):
    def __init__(self):
        super().__init__()