import sys
import os

from gym.core import ObservationWrapper
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig

import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback



# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
EXPERIMENT_NAME = "vivian_NO_CRASH_EMPTY"

logger_conf = CometLoggerConfig()
logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["Online_PPO"])





logger = CometLogger(logger_conf)
print(logger.log_dir)


########################################## real env setup ####################################
config = DefaultMainConfig()
config.populate_config(
    observation_config = "LowDimObservationConfig",
    action_config = "MergedSpeedScaledTanhConfig",
    reward_config = "Simple2RewardConfig",
    scenario_config = "NoCrashEmptyTown01Config",
    testing = False,
    carla_gpu = 0
)
# logger_callback = PPOLoggerCallback(logger)
env = CarlaEnv(config = config, logger = logger, log_dir = "/home/scratch/vccheng/carla_test")

#################################### fake env setup ########################################
fake_env_config = DefaultMainConfig()
fake_env_config.populate_config(\
    obs_config = "DefaultObservationConfig", \
    action_config = "DefaultActionConfig",\
    reward_config="DefaultRewardConfig",\
    uncertainty_config="DefaultUncertaintyConfig")
fake_env = FakeEnv(dynamics,
                config=fake_env_config,
                logger = logger,
                uncertainty_threshold = 0.5,
                uncertain_penalty = -100,
                timeout_steps = 1,
                uncertainty_params = [0.0045574815320799725, 1.9688976602303934e-05, 0.2866033549975823])


tb_log_dir = os.path.join(logger.log_dir, "ppo_tb_logs")
os.makedirs(tb_log_dir)

################################# instantiate model ######################################

model = Morel(env, fake_env, 
            action_dim,
            uncertainty_threshold,
            uncertainty_penalty,
            dynamics_epochs,
            policy_epochs)

model.train()

