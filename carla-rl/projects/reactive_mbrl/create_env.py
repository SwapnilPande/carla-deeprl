from environment import CarlaEnv, CarlaEvalEnv
import math
import importlib

import environment.config.scenario_configs
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.action_configs import *

# CAMERA_YAWS = [0, -30, 30]
CAMERA_YAWS = [0]
CAMERA_X = 1.5
CAMERA_Z = 2.4


def populate_cameras_configs(cfg):
    # for i, yaw in enumerate(CAMERA_YAWS):
    x = CAMERA_X * math.cos(0 * math.pi / 180)
    y = CAMERA_X * math.sin(0 * math.pi / 180)
    cfg["sensor.camera.rgb/top"] = {
        "x": 13.0,
        "z": 18.0,
        "pitch": 270,
        "sensor_x_res": "256",
        "sensor_y_res": "256",
        "fov": "90",
        "sensor_tick": "0.0",
    }
    
    cfg["sensor.camera.rgb/map"] = {
        "x": 13.0,
        "z": 18.0,
        "pitch": 270,
        "sensor_x_res": "16",
        "sensor_y_res": "16",
        "fov": "90",
        "sensor_tick": "0.0",
    }

    cfg["sensor.camera.rgb/front_wide"] = {
        "x": x,
        "y": y,
        "z": CAMERA_Z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": "60",
        "sensor_x_res": "480",
        "sensor_y_res": "240",
        "sensor_tick": "0.0",
    }
    
    cfg["sensor.camera.semantic_segmentation/front_wide"] = {
        "x": x,
        "y": y,
        "z": CAMERA_Z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": "60",
        "sensor_x_res": "480",
        "sensor_y_res": "240",
        "sensor_tick": "0.0",
        "num_classes": 23,
        "seg_channels": [4, 6, 7, 10, 18],
    }
    
    cfg["sensor.camera.rgb/front_narrow"] = {
        "x": x,
        "y": y,
        "z": CAMERA_Z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": "50",
        "sensor_x_res": "384",
        "sensor_y_res": "240",
        "sensor_tick": "0.0",
    }
    
    cfg["sensor.camera.semantic_segmentation/front_narrow"] = {
        "x": x,
        "y": y,
        "z": CAMERA_Z,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "fov": "50",
        "sensor_x_res": "384",
        "sensor_y_res": "240",
        "sensor_tick": "0.0",
        "num_classes": 23,
        "seg_channels": [4, 6, 7, 10, 18],
    }

    return cfg


def populate_other_sensors_configs(cfg):

    cfg["sensor.other.gnss"] = {
        "x": 0.0,
        "y": 0.0,
        "z": CAMERA_Z,
    }

    return cfg


def create_env_config(cfg):
    config = DefaultMainConfig()
    config.server_fps = 20

    obs_config = LowDimObservationConfig()
    obs_config.sensors = populate_cameras_configs(obs_config.sensors)
    obs_config.sensors = populate_other_sensors_configs(obs_config.sensors)

    scenario_config = create_scenario_config(cfg)

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(
        observation_config=obs_config, scenario_config=scenario_config
    )
    return config

def create_scenario_config(cfg):
    class_name = f"{cfg.family}{cfg.town}Config"
    class_ = getattr(environment.config.scenario_configs, class_name)
    return class_()


def create_env(run_config, log_dir):
    config = create_env_config(run_config)
    env = CarlaEnv(config=config, log_dir=log_dir + "/")
    return env


def create_eval_env(log_dir):
    config = create_env_config()
    env = CarlaEnv(config=config, log_dir=log_dir + "/")
    return CarlaEvalEnv(env, 5000)
