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
        self.gpu = gpu

class BasePolicyConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.learning_rate = None

        # Discount Factor
        self.gamma = None


class BasePPOConfig(BasePolicyConfig):
    def __init__(self):
        super().__init__()

        # # Clip range for PPO
        # self.clip_range = None

        # # Number of epochs to train for policy updates
        # self.n_epochs = None


class DefaultPPOConfig(BasePPOConfig):
    def __init__(self):
        super().__init__()

        self.learning_rate = 3e-4
        self.gamma = 0.99
        # self.clip_range = 0.25
        # self.n_epochs = 10

class BaseMOPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        ## Config for the dynamics model
        # If pretrained dynamics model is not None, we will load it from comet
        self.pretrained_dynamics_model = None
        # Else, we will train a new dynamics model using the dynamics_config passed
        self.dynamics_config = None

        self.fake_env_config = None

        self.eval_env_config = None

        self.policy_algorithm = None

        self.policy_hyperparameters = None

    def populate_config(self, gpu = 0, policy_algorithm = "PPO", pretrained_dynamics_model_key = None, pretrained_dynamics_model_name = None):
        self.gpu = gpu

        # Setup dynamics config
        # Use pretrained model if available. Else, use the dynamics_config passed
        if(pretrained_dynamics_model_key is not None):
            # self.dynamics_config = DefaultMLPObstaclesMOPOConfig()
            self.pretrained_dynamics_model = PretrainedDynamicsModelConfig()
            self.pretrained_dynamics_model.populate(key = pretrained_dynamics_model_key,
                                                    name = pretrained_dynamics_model_name,
                                                    gpu = gpu)

            self.dynamics_config = None

        else:
            self.dynamics_config.populate_config(gpu = gpu)


        self.eval_env_config.carla_gpu = gpu

        self.policy_algorithm = getattr(stable_baselines3, policy_algorithm)
        if(policy_algorithm == "PPO"):
            self.policy_hyperparameters = DefaultPPOConfig()
        else:
            raise Exception("No config for policy algorithm: {}".format(policy_algorithm))

        if(pretrained_dynamics_model_key is not None):
            ignore_keys = ["dynamics_config"]
        else:
            ignore_keys = ["pretrained_dynamics_model_config"]
        self.verify(ignore_keys = ignore_keys)


class DefaultMLPMOPOConfig(BaseMOPOConfig):
    def __init__(self):
        super().__init__()

        self.dynamics_config = dynamics_config.DefaultMLPDynamicsConfig()

        self.fake_env_config = fake_env_config.NoTimeoutFakeEnvConfig()

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

        self.fake_env_config = fake_env_config.NoTimeoutFakeEnvConfig()

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
            carla_gpu = self.gpu
        )
        # Disable traffic lights
        self.eval_env_config.scenario_config.set_parameter("disable_traffic_light", True)
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

