import sys
import os

from gym.core import ObservationWrapper
# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig
from projects.morel_mopo.algorithm.morel import Morel
import gym
from algorithms import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import DummyVecEnv
#from common.loggers.logger_callbacks import PPOLoggerCallback



# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig
EXPERIMENT_NAME = "vivian_NO_CRASH_EMPTY"

########################################## logger  ####################################

# logger_conf = CometLoggerConfig()
# logger_conf.populate(experiment_name = EXPERIMENT_NAME, tags = ["Online_PPO"])
# logger = CometLogger(logger_conf)
# print(logger.log_dir)

# tb_log_dir = os.path.join(logger.log_dir, "ppo_tb_logs")
# os.makedirs(tb_log_dir)

################################# pass in envs and instantiate model ######################################
uncertainty_threshold = 0.5
uncertainty_penalty = -100
dynamics_epochs = 10
policy_epochs = 10

model = Morel(uncertainty_threshold,
            uncertainty_penalty,
            dynamics_epochs,
            policy_epochs,
            logger=None)#logger)

# train MOReL
model.train()

