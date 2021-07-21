from environment.config.base_config import BaseConfig
from environment.config.config import DefaultMainConfig


from projects.morel_mopo.config import dynamics_config
from projects.morel_mopo.config import fake_env_config

class BaseMOPOConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.gpu = None

        # Config for the dynamics model
        self.dynamics_config = None

        self.fake_env_config = None

        self.eval_env_config = None

    def populate_config(self, gpu = 0):
        self.gpu = 0
        self.dynamics_config.populate_config(gpu = gpu)
        self.eval_env_config.carla_gpu = gpu

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

