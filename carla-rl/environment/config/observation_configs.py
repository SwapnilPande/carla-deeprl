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
                        "sensor.camera.rgb/top": {'x':3.0,
                                                                    'z':20.0,
                                                                    'pitch':270.0,
                                                                    'sensor_x_res': '512',
                                                                    'sensor_y_res':'512',
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

class VehicleDynamicsObstacleNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obstacle_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
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
        self.disable_lane_invasion_sensor = True

class VehicleDynamicsObstacleConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obstacle_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
                            "sensor.camera.rgb/top": {'x': 7,
                                                    'z':25,
                                                    'pitch': -90.0,
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

class VehicleDynamicsObstacleLightNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obstacle_speed_steer_light"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, 0.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
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

class VehicleDynamicsObstacleLightConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obstacle_speed_steer_light"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, 0.0, 0.0, 0.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
                            "sensor.camera.rgb/top": {'x': 5,
                                                    'z':10,
                                                    'pitch': -90.0,
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
        self.disable_lane_invasion_sensor = True#False


class VehicleDynamicsConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
                            "sensor.camera.rgb/top": {'x':0,
                                                    'z':10,
                                                    'pitch': -90.0,
                                                    'sensor_x_res':'1080',
                                                    'sensor_y_res':'1920',
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

class VehicleDynamicsNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
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



class VehicleDynamicsExtendedLookaheadConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_extended_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, -4.0, 0.0, -0.5, -1.0]]),
                            high=np.array([[4.0, 4.0, 1.0, 0.5, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
                            "sensor.camera.rgb/top": {'x':0,
                                                    'z':10,
                                                    'pitch': -90.0,
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

class VehicleDynamicsExtendedLookaheadNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_obs_info_extended_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, -4.0, 0.0, -0.5, -1.0]]),
                            high=np.array([[4.0, 4.0, 1.0, 0.5, 1.0]]),
                            dtype=np.float32)


        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
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


class LeaderboardObsNoCameraConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_360_obstacle_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                            dtype=np.float32)

        hit_radius = 2
        self.sensors = {
                            "collision_sensor": None,
                            "obstacle_sensor_0" : {
                                "x" : 0,
                                "y" : 0.0,
                                "z" : 0.3,
                                "yaw" : 0,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_1" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 12,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_2" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 24,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_3" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 36,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_4" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 48,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_5" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 60,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_6" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 72,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_7" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 84,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_8" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 96,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_9" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 108,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_10" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 120,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_11" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 132,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_12" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 144,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_13" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 156,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_14" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 168,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_15" : {
                                "x" : -1.5,
                                "y" : 0,
                                "z" : 0.3,
                                "yaw" : 180,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_16" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 192,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_17" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 204,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_18" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 216,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_19" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 228,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_20" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 240,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_21" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 252,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_22" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 264,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_23" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 276,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_24" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 288,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_25" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 300,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_26" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 312,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_27" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 324,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_28" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 336,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_29" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 348,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_30" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 5,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_31" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : -5,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_32" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 175,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_33" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : -175,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                    }
        self.observation_sensors = []

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1.5
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 45
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = True

class LeaderboardObsConfig(BaseObservationConfig):
    def __init__(self):
        self.input_type = "wp_360_obstacle_speed_steer"
        self.observation_space = Box(low=np.array([[-4.0, 0.0, -0.5, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]),
                            high=np.array([[4.0, 1.0, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]]),
                            dtype=np.float32)


        hit_radius = 2
        self.sensors = {
                            "lane_invasion_sensor":None,
                            "collision_sensor": None,
                            "obstacle_sensor_0" : {
                                "x" : 0,
                                "y" : 0.0,
                                "z" : 0.3,
                                "yaw" : 0,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_1" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 12,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_2" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 24,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_3" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 36,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_4" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 48,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_5" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 60,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_6" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 72,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_7" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 84,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_8" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 96,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_9" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 108,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_10" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 120,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_11" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 132,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_12" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 144,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_13" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 156,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_14" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 168,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_15" : {
                                "x" : -1.5,
                                "y" : 0,
                                "z" : 0.3,
                                "yaw" : 180,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_16" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 192,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_17" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 204,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_18" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 216,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_19" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 228,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_20" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 240,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_21" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 252,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_22" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 264,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_23" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 276,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_24" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 288,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_25" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 300,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_26" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 312,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_27" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 324,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_28" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 336,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_29" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : 348,
                                "distance" : 45,
                                "hit_radius" : hit_radius,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_30" : {
                                "x" : 1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 5,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_31" : {
                                "x" : 1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : -5,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_32" : {
                                "x" : -1.5,
                                "y" : 1.0,
                                "z" : 0.3,
                                "yaw" : 175,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "obstacle_sensor_33" : {
                                "x" : -1.5,
                                "y" : -1.0,
                                "z" : 0.3,
                                "yaw" : -175,
                                "distance" : 45,
                                "hit_radius" : 0.1,
                                'only_dynamics' : True,
                            },
                            "sensor.camera.rgb/top": {'x':5,
                                                    'z':40,
                                                    'pitch': -90.0,
                                                    'sensor_x_res':'256',
                                                    'sensor_y_res':'256',
                                                    'fov':'90',
                                                    'sensor_tick': '0.0'
                            }
                    }
        self.observation_sensors = []

        self.single_channel_image = False
        self.noise_dim = 1
        self.preprocess_crop_image = True
        self.grayscale = False
        self.default_obs_traffic_val = 1.5
        self.min_dist_from_red_light = 4
        self.disable_obstacle_info = False
        self.frame_stack_size = 1
        self.vehicle_proximity_threshold = 45
        self.traffic_light_proximity_threshold = 15
        self.obstacle_dist_norm = 60
        self.disable_lane_invasion_sensor = True


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
        self.observation_sensors = ['sensor.camera.rgb/front', "sensor.camera.semantic_segmentation/top"]

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
