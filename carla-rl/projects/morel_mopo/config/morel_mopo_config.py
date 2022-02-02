from environment.config.base_config import BaseConfig
from environment.config.config import DefaultMainConfig


from projects.morel_mopo.config import dynamics_config
from projects.morel_mopo.config import fake_env_config

import stable_baselines3


class PretrainedDynamicsModelConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.key = None
        self.name = None
        self.gpu = None

    def populate(self, key, name, gpu):
        self.key = key
        self.name = name

class BaseMOPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        ## Config for the dynamics model
        # If pretrained dynamics model is not None, we will load it from comet
        self.pretrianed_dynamics_model = None
        # Else, we will train a new dynamics model using the dynamics_config passed
        self.dynamics_config = None

        self.fake_env_config = None

        self.eval_env_config = None

        self.policy_algorithm = None

    def populate_config(self, gpu = 0, policy_algorithm = "PPO", pretrained_dynamics_model_key = None, pretrained_dynamics_model_name = None):
        self.gpu = 0

        # Setup dynamics config
        # Use pretrained model if available. Else, use the dynamics_config passed
        if(pretrained_dynamics_model_key is not None):
            self.dynamics_config = NoneDefaultMLPObstaclesMOPOConfig
            self.pretrained_dynamics_model = PretrainedDynamicsModelConfig()
            self.pretrained_dynamics_model.populate(key = pretrained_dynamics_model_key,
                                                    name = pretrained_dynamics_model_name)

        else:
            self.dynamics_config.populate_config(gpu = gpu)


        self.eval_env_config.carla_gpu = gpu

        self.policy_algorithm = getattr(stable_baselines3, policy_algorithm)

        self.verify()


class DefaultMLPMOPOConfig(BaseMOPOConfig):
    def __init__(self):
        super().__init__()

        self.dynamics_config = dynamics_config.DefaultMLPDynamicsConfig()

        self.fake_env_config = fake_env_config.DefaultFakeEnvConfig()

        self.fake_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config="Simple2RewardConfig"
        )

        self.eval_env_config = DefaultMainConfig()
        self.eval_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashEmptyTown01Config",
            testing = False,
            carla_gpu = self.gpu
        )

class DefaultMLPObstaclesMOPOConfig(BaseMOPOConfig):
    def __init__(self):
        super().__init__()

        self.dynamics_config = dynamics_config.ObstaclesMLPDynamicsConfig()

        self.fake_env_config = fake_env_config.DefaultFakeEnvConfig()

        self.fake_env_config.populate_config(
            observation_config = "VehicleDynamicsObstacleNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config="Simple2RewardConfig"
        )

        self.eval_env_config = DefaultMainConfig()
        self.eval_env_config.populate_config(
            observation_config = "VehicleDynamicsObstacleNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashDenseTown01Config",
            testing = False,
            carla_gpu = self.gpu
        )

class DefaultProbMLPMOPOConfig(BaseMOPOConfig):
    def __init__(self):
        super().__init__()

        self.dynamics_config = dynamics_config.DefaultProbabilisticMLPDynamicsConfig()

        self.fake_env_config = fake_env_config.DefaultFakeEnvConfig()

        self.fake_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config="Simple2RewardConfig"
        )

        self.eval_env_config = DefaultMainConfig()
        self.eval_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashEmptyTown01Config",
            testing = False,
            carla_gpu = self.gpu
        )

class DefaultProbGRUMOPOConfig(BaseMOPOConfig):
    def __init__(self):
        super().__init__()

        self.dynamics_config = dynamics_config.DefaultProbabilisticGRUDynamicsConfig()

        self.fake_env_config = fake_env_config.DefaultFakeEnvConfig()

        self.fake_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config="Simple2RewardConfig"
        )

        self.eval_env_config = DefaultMainConfig()
        self.eval_env_config.populate_config(
            observation_config = "VehicleDynamicsNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashEmptyTown01Config",
            testing = False,
            carla_gpu = self.gpu
        )

