from environment.config.base_config import BaseConfig
from environment.config.config import DefaultMainConfig


from projects.morel_mopo.config import dynamics_config
from projects.morel_mopo.config import fake_env_config

import stable_baselines3

class BaseMOPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Config for the dynamics model
        self.dynamics_config = None

        self.fake_env_config = None

        self.eval_env_config = None

        self.policy_algorithm = None

    def populate_config(self, gpu = 0, policy_algorithm = "PPO"):
        self.gpu = 0
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
            action_config = "MergedSpeedTanhConfig",
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
            action_config = "MergedSpeedTanhConfig",
            reward_config="Simple2RewardConfig"
        )

        self.eval_env_config = DefaultMainConfig()
        self.eval_env_config.populate_config(
            observation_config = "VehicleDynamicsObstacleNoCameraConfig",
            action_config = "MergedSpeedScaledTanhConfig",
            reward_config = "Simple2RewardConfig",
            scenario_config = "NoCrashRegularTown01Config",
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
            action_config = "MergedSpeedTanhConfig",
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
            action_config = "MergedSpeedTanhConfig",
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

