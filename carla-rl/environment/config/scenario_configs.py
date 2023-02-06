from environment.config.base_config import BaseConfig


class BaseScenarioConfig(BaseConfig):
    def __init__(self):
        # Whether to use Dynamic Navigation/NoCrash scenarios
        # If False, simply select the random source and destination points
        self.use_scenarios = None

        # Which scenario to use
        # Valid Options: straight, long_straight, long_straight_junction, straight_dynamic
        # crowded, straight_crowded, town3, left_right_curved, right_curved, left_curved
        # t_junction, curved, navigation, dynamic_navigation, no_crash_empty, no_crash_regular,
        # no_crash_dense, challenge_train_scenario, challenge_test_scenario
        self.scenarios = None

        # If true, will sample random number of cars within the range
        #    [num_npc_lower_threshold, num_npc_upper_threshold)
        # Else, will spawn num_npc number of vehicles
        self.sample_npc = None
        self.num_npc = None
        self.num_npc_lower_threshold = None
        self.num_npc_upper_threshold = None

        # Number of pedestrians to spawn
        self.num_pedestrians = None
        # Should two wheelers be allowed
        self.disable_two_wheeler = None

        # City to load
        # Town to load TODO, leaderboard doesn't use this, is there a better way to handle this?
        self.city_name = None

        # Ego vehicle make/model
        self.vehicle_type = None

        # Threshold distance from target transform to consider episode a success
        self.dist_for_success = None

        # Maximum length of episode, episode is terminated if this is exceeded
        self.max_steps = None

        # Maximum number of steps the vehicle is allowed to be offlane.
        # If offlane steps is exceeded, episode is terminated
        self.max_offlane_steps = None

        # Maximum number of steps vehicle is allowed to be static
        # If this is exceeded, episode is terminated
        self.max_static_steps = None

        #TODO add comments describing these
        self.disable_collision = None
        self.disable_static = None
        self.disable_traffic_light = None
        # self.disable_lane_invasion_sensor = None
        self.zero_speed_threshold = None

        # Number of episodes to run the scenarios for
        self.num_episodes = None

        # FIGURE OUT WHAT THIS IS
        self.updated_scenarios = None

        # Whether to count a lane invasion as a collision
        self.disable_lane_invasion_collision = None

class DynamicNavigationConfig(BaseScenarioConfig):
    def __init__(self):
        super().__init__()

        self.use_scenarios = True
        self.scenarios = "dynamic_navigation"
        self.sample_npc = None
        self.num_npc = None
        self.num_npc_lower_threshold = None
        self.num_npc_upper_threshold = None
        self.num_pedestrians = None
        self.disable_two_wheeler = None
        self.city_name = None
        self.vehicle_type = None
        self.dist_for_success = None
        self.max_steps = None
        self.max_offlane_steps = None
        self.max_static_steps = None
        self.disable_collision = None
        self.disable_static = None
        self.disable_traffic_light = None
        self.disable_lane_invasion_sensor = None
        self.zero_speed_threshold = None
        self.num_episodes = 25


class NoCrashConfig(BaseScenarioConfig):
    def __init__(self):
        super().__init__()
        self.use_scenarios = True
        self.sample_npc = False
        self.num_npc_lower_threshold = 0
        self.num_npc_upper_threshold = 0
        self.num_pedestrians = 0
        self.disable_two_wheeler = True
        #TODO Make this flexible for training/testing
        self.vehicle_type = 'vehicle.toyota.prius'
        self.dist_for_success = 10.0
        self.max_steps = 10000
        self.max_offlane_steps = 20
        self.max_static_steps = 1000

        self.disable_collision = False
        # Disable episode termination due to vehicle being static
        self.disable_static = False
        # Disable episode termination due to traffic light
        self.disable_traffic_light = False
        self.zero_speed_threshold = 0.05
        self.num_episodes = 25
        self.updated_scenarios = False
        self.disable_lane_invasion_collision = True
        self.disable_lane_invasion_sensor = True


class NoCrashEmptyTown01Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_empty"
        self.num_npc = 0
        self.city_name = "Town01"

class NoCrashEmptyTown02Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_empty"
        self.num_npc = 0
        self.city_name = "Town02"

class NoCrashRegularTown01Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_regular"
        self.num_npc = 20
        self.city_name = "Town01"

class NoCrashRegularTown02Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_regular"
        self.num_npc = 15
        self.city_name = "Town02"

class NoCrashDenseTown01Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_dense"
        self.num_npc = 100
        self.city_name = "Town01"

class NoCrashDenseTown02Config(NoCrashConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "no_crash_dense"
        self.num_npc = 70
        self.city_name = "Town02"

class LeaderboardConfig(BaseScenarioConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "challenge_train_scenario"
        self.use_scenarios = True
        self.sample_npc = False
        self.num_npc = 50
        self.num_npc_lower_threshold = 60
        self.num_npc_upper_threshold = 80
        self.num_pedestrians = 200
        self.disable_two_wheeler = False
        self.city_name = 'Town06'
        self.vehicle_type = 'vehicle.toyota.prius'
        self.min_num_eps_before_switch_town = 3

        self.city_name = "Town01"
        self.use_scenarios = True
        self.scenarios = "challenge_train_scenario"
        self.updated_scenarios = False
        self.vehicle_type = "vehicle.toyota.prius"
        self.num_pedestrians = 0
        self.num_npc = 0
        self.sample_npc = True
        self.num_npc_lower_threshold = 70
        self.num_npc_upper_threshold = 150
        self.traffic_light_proximity_threshold = 15
        self.min_dist_from_red_light = 4
        self.disable_traffic_light = False
        self.disable_collision = False
        self.dist_for_success = 10
        self.zero_speed_threshold = 0.05
        self.max_offlane_steps = 20
        self.max_steps = 10000
        self.max_static_steps = 1000
        self.disable_two_wheeler = True
        self.enable_lane_invasion_sensor = True
        self.disable_static = True
        self.disable_lane_invasion_collision = False


class NoTrafficLightNoCrashEmptyConfig(NoCrashEmptyTown01Config):
    def __init__(self):
        super().__init__()
        self.disable_traffic_light = True

class NoTrafficLightNoCrashDenseConfig(NoCrashDenseTown01Config):
    def __init__(self):
        super().__init__()
        self.disable_traffic_light = True

class SimpleSingleTurnConfig(NoTrafficLightNoCrashEmptyConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "simple_single_turn"

class SimpleStraightPathConfig(NoTrafficLightNoCrashEmptyConfig):
    def __init__(self):
        super().__init__()
        self.scenarios = "simple_straight_path"
