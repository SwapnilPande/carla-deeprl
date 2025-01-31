from environment.config.base_config import BaseConfig
import numpy as np
from gym.spaces import Box

class BaseObservationConfig(BaseConfig):
    def __init__(self):
        # Name of observation type
        self.input_type = None

        # Gym Observation Space
        self.obs_space = None

        # Key is sensor name, value is configuration parameters
        self.sensors = None
        self.observation_sensors = None
        self.single_channel_image = None
        self.noise_dim = None
        self.preprocess_crop_image = None
        self.grayscale = None
        self.default_obs_traffic_val = None
        self.min_dist_from_red_light = None

        self.disable_obstacle_info = None
        # Number of frames to stack together in observation
        self.frame_stack_size = None

        # Threshold for maximum distance for recognizing other vehicles as being nearby
        self.vehicle_proximity_threshold = None

        # Threshold for maximum distance for recgonizing traffic light as being nearby
        self.traffic_light_proximity_threshold = None

        # Normalization Constant for obstacle measurement
        self.obstacle_dist_norm = None

        # Whether or not the lane invasion sensor is enabled
        # TODO These are redundant parameters, with the "sensors" parameter
        # TODO We need to remove these
        self.disable_lane_invasion_sensor = None


class LowDimObservationConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer_ldist_goal_light"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {"lane_invasion_sensor":None, \
                        "collision_sensor": None, \
                        "sensor.camera.semantic_segmentation/top": {'x':3.0,
                                                                    'z':10.0,
                                                                    'pitch':270.0,
                                                                    'sensor_x_res': '128',
                                                                    'sensor_y_res':'128',
                                                                    'fov':'90',
                                                                    'sensor_tick': '0.0',
                                                                    'num_classes':5},
                        "sensor.camera.rgb/front": {'x':2.0,
                                                    'z':1.4,
                                                    'pitch':0.0,
                                                    'sensor_x_res':'112',
                                                    'sensor_y_res':'112',
                                                    'fov':'90',
                                                    'sensor_tick': '0.0'}
                        }
        self.observation_sensors = []

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 15
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = False

class LowDimObservationNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer_ldist_goal_light"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None
                    }
        self.observation_sensors = []

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 15
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = False

class VehicleDynamicsNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None
                    }
        self.observation_sensors = []

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 15
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = False


class PerspectiveRGBObservationConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer_ldist_goal_light"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, 0.0, 0.0, -0.5, -1.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {"lane_invasion_sensor":None, \
                        "collision_sensor": None, \
                        "sensor.camera.semantic_segmentation/top": {'x':3.0,
                                                                    'z':10.0,
                                                                    'pitch':270.0,
                                                                    'sensor_x_res': '128',
                                                                    'sensor_y_res':'128',
                                                                    'fov':'90',
                                                                    'sensor_tick': '0.0',
                                                                    'num_classes':5},
                        "sensor.camera.rgb/front": {'x':2.0,
                                                    'z':1.4,
                                                    'pitch':0.0,
                                                    'sensor_x_res':'112',
                                                    'sensor_y_res':'112',
                                                    'fov':'90',
                                                    'sensor_tick': '0.0'}
                        }
        self.observation_sensors = ['sensor.camera.rgb/front']

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 15
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = False
