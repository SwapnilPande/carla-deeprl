import os
import sys
import glob

CARLA_9_4_PATH = os.environ.get("CARLA_9_4_PATH")
if CARLA_9_4_PATH == None:
    raise ValueError("Set $CARLA_9_4_PATH to directory that contains CarlaUE4.sh")

try:
    sys.path.append(glob.glob(CARLA_9_4_PATH+ '/**/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print(".egg file not found! Kindly check for your Carla installation.")
    pass

DEFAULT_ENV = {
    "server_path" : CARLA_9_4_PATH,
    "server_binary" : CARLA_9_4_PATH + '/CarlaUE4.sh',
    "server_process" : None,
    # X Rendering Resolution
    "render_res_x" : 800,
    # Y Rendering Resolution
    "render_res_y" : 800,
    "sensor_x_res" : '128',
    "sensor_y_res" : '128',
    # Input X Res (Default set to Atari)
    "x_res": 84,
    # Input Y Res (Default set to Atari)
    "y_res": 84,
    "server_fps" : 10,
    "server_port" : None,
    "server_retries" : 5,
    "city_name" : "Town01",
    "frame_skip": 1,
    "enable_planner" : True,
    "reward_function" : 'corl',
    # Print measurements to screen
    "print_obs" : True,
    "client" : None,
    "discrete_actions": False,

    # Number of frames stacked together
    "framestack" : 1,
    "grayscale" : False,
    "num_pedestrians" : 0,
    "max_steps" : 10000,
    "next_command": None,
    "verbose": False,
    "vehicle_type": 'vehicle.toyota.prius',
    "disable_two_wheeler" : True,
    "vehicle_types": ['vehicle.ford.mustang', 'vehicle.audi.a2', 'vehicle.audi.tt', 'vehicle.bmw.isetta', 'vehicle.carlamotors.carlacola',
                      'vehicle.citroen.c3', 'vehicle.bmw.grandtourer', 'vehicle.mercedes-benz.coupe',
                      'vehicle.toyota.prius', 'vehicle.dodge_charger.police', 'vehicle.nissan.patrol',
                      'vehicle.tesla.model3', 'vehicle.seat.leon', 'vehicle.lincoln.mkz2017',
                      'vehicle.volkswagen.t2', 'vehicle.nissan.micra', 'vehicle.chevrolet.impala', 'vehicle.mini.cooperst',
                      'vehicle.jeep.wrangler_rubicon'],
    "target_speed": 20,
    "sensors": ["sensor.camera.rgb", "sensor.camera.semantic_segmentation"],
    "action_type": "merged_gas",
    "sensor_tick": '0.0',
    "dist_for_success" : 10.0,
    "max_offlane_steps" : 20,
    "max_static_steps" : 1000,
    "log_measurements_to_file": False,
    "train_config": "PPO",
    "sync_mode": True,
    # NOTE: crop does not work with framestack yet. need to add.
    "preprocess_crop_image": False,
    "scenarios" : "straight",
    "semantic" : False,
    "client_timeout_seconds" : 10,
    "enable_lane_invasion_sensor" : True,
    "carla_gpu": "0",
    "render_server": True,
    "steer_penalty_coeff": 0,
    "vae_encoding_norm_factor" : 10,
    "input_type": None,
    "use_scenarios": True,
    "num_npc" : 0,
    "num_npc_lower_threshold" : 70,
    "num_npc_upper_threshold" : 150,
    "train_vae" : False,
    "binarized_image": False,
    "single_channel_image": False,
    "noise_dim" : 1,
    "const_collision_penalty": 0,
    "collision_penalty_speed_coeff": 0,
    "const_light_penalty": 0,
    "light_penalty_speed_coeff": 0,
    "terminate_on_light" : False,
    "enable_brake": True,
    "log_freq": 1,
    "zero_speed_threshold": 0.05,
    "videos" : False,
    "obstacle_dist_norm" : 60,
    "spawn_points_fixed_idx" : [ 54, 234, 108,  12, 175,  71, 116,  99, 196,  63, 205,  46,  96,
       246, 128, 106, 143,  39,  72, 176, 140, 138,  91,  88, 241,  29,
        28, 238, 119, 221, 163,  81,  47, 255, 235,  64, 216, 151, 145,
        77,  35,  56,  68,  49, 154, 149, 201,  27, 212, 195, 230, 157,
         3,   5,  20, 193,   6,  90,  18,  13, 139,  44, 122, 220, 125,
       115,  43,   4, 213,  30,  62, 242, 219, 171,  41, 203,  57, 248,
       204, 226, 245, 135, 164, 153,  14, 188,   7, 123, 117, 222, 183,
       152, 150, 185, 224,  19, 104, 111,  82,  79,   0,  33,  38, 146,
        10, 173, 239,  32, 228, 209, 243, 200, 215, 236,  34,  84,  51,
        73,  53, 170, 217, 237, 102, 156,  45, 253,  37, 210, 118,  86,
        74,  61, 165, 179, 202, 101,  36, 132, 168, 137, 126, 178,  24,
         1, 247, 107,  93, 148,  50,  98,  87, 133, 162,   2, 214, 124,
       112, 211,  75, 121, 191, 113, 141,  26, 231, 174,  76, 207, 109,
       244, 129, 103,  52,  42,  55, 180,  89, 181,  69,  48,  21,  16,
       198,  66,  70, 130, 114,  15, 134,  40, 227, 223,  67,  78, 159,
       252, 147,  17, 166,  11, 131, 161, 105, 167,  95, 172, 233, 251,
       194,  60,  80, 182,  97,  59, 197,  25, 186, 136, 160, 120, 158,
       189, 192, 190, 187, 142, 232,   9, 127, 206, 169,  23, 208,  94,
       218,  83, 155,  65, 254, 249,  92, 240,  85, 100,  58,  22,   8,
       225,  31, 229, 250, 110, 177, 199, 184, 144],
    "test_fixed_spawn_points" : True,
    "train_fixed_spawn_points": False,
    "testing" : False,
    "disable_collision": False,
    "enable_static": False,
    "use_pid_in_frame_skip" : True,
    "enable_lane_invasion_collision" : True,
    "vehicle_proximity_threshold" : 15,
    "traffic_light_proximity_threshold" : 10,
    "min_dist_from_red_light" : 4,
    "clip_reward" : False,
    "default_obs_traffic_val": 1,
    "reward_normalize_factor": 1,
    "success_reward": 0,
    "constant_positive_reward": 0,
    "frame_stack_size" : 1,
    "num_episodes" : 1,
    "disable_traffic_light": False,
    "disable_obstacle_info" : False,
    "test_comparison": False,
    "test_with_automatic_control": False,
    "updated_scenarios": False,
    "sample_npc": True,
    "use_offline_map": False,
    "map_path" : "/home/hitesh/research/repos/alta/environment/carla_9_4/OpenDrive/Town01.xodr",
    "use_route_to_plan" : False,
    "min_num_eps_before_switch_town": 3,
}

episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "speed": None,
    "distance_to_goal": None,
    "num_collisions": 0,
    "num_laneintersections": 0,
    "static_steps": 0,
    "offlane_steps": 0,
    "control_steer": 0
    # intersection_offroad
    # intersection_otherlane
    # next_command
}

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [0.5, 0.0],
#     # Forward left
#     2: [0.25, -0.3],
#     3: [0.25, -0.1],
#     # Forward right
#     4: [0.25, 0.1],
#     5: [0.25, 0.3],
#     # Brake
#     6: [-0.5, 0.0],
#     # Brake left
#     7: [-0.25, -0.3],
#     8: [-0.25, -0.1],
#     # Brake right
#     9: [-0.25, 0.1],
#     10: [-0.25, 0.3]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [10.0, 0.0],
#     # Forward
#     1: [20.0, 0.0],
#     # Forward left
#     2: [15.0, -0.3],
#     3: [15.0, -0.1],
#     # Forward right
#     4: [15.0, 0.1],
#     5: [15.0, 0.3],
#     # Brake
#     6: [0.0, 0.0],
#     # Brake left
#     7: [5.0, -0.3],
#     8: [5.0, -0.1],
#     # Brake right
#     9: [5.0, 0.1],
#     10: [5.0, 0.3]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [10.0, -0.5],
#     # Forward
#     1: [10.0, -0.4],
#     # Brake
#     2: [10.0, -0.3],
#     # Left
#     3: [10.0, -0.2],
#     # Right
#     4: [10.0, -0.1],
#     # Forward left
#     5: [10.0, 0.0],
#     # Forward right
#     6: [10.0, 0.1],
#     # Brake left
#     7: [10.0, 0.2],
#     # Brake right
#     8: [10.0, 0.3],
#     9: [10.0, 0.4],
#     10: [10.0, 0.5]
# }

# DISCRETE_ACTIONS = {
#     # Coast
#     0: [0.0, 0.0],
#     # Forward
#     1: [2.0, 0.0],
#     # Forward left
#     2: [4.0, 0.0],
#     3: [6.0, 0.0],
#     # Forward right
#     4: [8.0, 0.0],
#     5: [10.0, 0.0],
#     # Brake
#     6: [12.0, 0.0],
#     # Brake left
#     7: [14.0, 0.0],
#     8: [16.0, 0.0],
#     # Brake right
#     9: [18.0, 0.0],
#     10: [20.0, 0.0]
# }

def get_discrete_actions():
    # steer = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]
    steer = [-0.3, -0.1, 0.0, 0.1, 0.3]
    # steer = [-0.1, 0.0, 0.1]
    # steer = [0.0]
    # target_speed = [0, 10, 20]
    # target_speed = [20]
    target_speed = [0, 20]

    # Dictionary of discrete (Target_Speed, Steer) actions
    action_space = {}

    n = 0
    for i in range(len(target_speed)):
        for j in range(len(steer)):
            action_space[n] = [target_speed[i], steer[j]]
            n = n+1

    action_space[n] = [20, -0.5]
    action_space[n+1] = [20, 0.5]
    return action_space

DISCRETE_ACTIONS = get_discrete_actions()

class ConfigManager(object):
    def __init__(self, algo='PPO'):
        # self.config = {'client_timeout_seconds': 100,}
        self.config = DEFAULT_ENV

        self._initialize_config(algo)

    def _initialize_config(self, algo):
        if algo == 'DDPG':
            self.config["algo"] = "DDPG"
            self.config["x_res"] = 200
            self.config["y_res"] = 84
            self.config["reward_function"] = "cirl"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "torch"
            self.config["action_type"] = "merged_gas"
        elif algo == 'DQN':
            self.config["algo"] = "DQN"
            self.config["x_res"] = 84
            self.config["y_res"] = 84
            self.config["reward_function"] = "simple2"
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "discrete"
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["scenarios"] = "navigation"
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = False
            self.config["max_steps"] = 5000
        elif algo == 'PPO':
            self.config["algo"] = "PPO"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed_scaled_tanh"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            # self.config["semantic"] = True
            self.config["sensors"] = {"lane_invasion_sensor":None, \
                                        "collision_sensor": None, \
                                        "sensor.camera.semantic_segmentation/top": {'x':13.0, 'z':18.0, 'pitch':270.0, \
                                                                                    'sensor_x_res': '112', 'sensor_y_res':'112', 'fov':'90', \
                                                                                    'sensor_tick': '0.0', 'num_classes':5},
                                        "sensor.camera.rgb/front": {'x':2.0, 'z':1.4, 'pitch':0.0, \
                                                                    'sensor_x_res':'112', 'sensor_y_res':'112', 'fov':'90', \
                                                                    'sensor_tick': '0.0'} }
            # self.config["scenarios"] = "navigation"
            self.config["scenarios"] = "challenge_train_scenario"
            self.config["videos"] = False
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            # self.config["input_type"] = "wp_bev_rv_obs_info_speed_steer_ldist_goal_light"
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = False
            self.config["carla_gpu"] = "0"
            self.config["disable_two_wheeler"] = True
            self.config["enable_lane_invasion_sensor"] = True
            # self.config["traffic_light_proximity_threshold"] = 15
            # self.config["min_dist_from_red_light"] = 6
            self.config["sample_npc"] = True
        elif algo == 'SAC':
            self.config["algo"] = "SAC"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["semantic"] = True
            self.config["scenarios"] = "straight"
            self.config["videos"] = True
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = True
            self.config["carla_gpu"] = "1"
        elif algo == 'AE':
            self.config["algo"] = "AE"
            self.config["action_type"] = "control"
            self.config["use_scenarios"] = False
            self.config["semantic"] = True
            self.config['max_steps'] = 10000
            self.config["city_name"] = "Town01"
            self.config["num_npc"] = 60
            self.config["input_type"] = "ae_train"
            self.config["videos"] = True
        elif algo == 'PID_TUNE':
            self.config["algo"] = "PPO"
            self.config["reward_function"] = "simple2"
            self.config["discrete_actions"] = False
            self.config["train_config"] = "PPO"
            self.config["action_type"] = "merged_speed_pid_test"
            self.config["preprocess_crop_image"] = True
            self.config["framestack"] = 1
            self.config["grayscale"] = False
            self.config["semantic"] = False
            self.config["scenarios"] = "straight_dynamic"
            self.config["videos"] = False
            self.config["x_res"] = 80
            self.config["y_res"] = 160
            self.config["input_type"] = "wp"
            self.config["city_name"] = "Town01"
            self.config["verbose"] = True
            self.config["carla_gpu"] = "1"
            self.config["max_static_steps"] = 20
