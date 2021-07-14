import environment.carla_interfaces.scenarios_910 as scenarios
from environment.carla_interfaces.server import CarlaServer
from environment.carla_interfaces import planner
from environment.carla_interfaces.actor_manager import ActorManager910, ActorManager910_Leaderboard
from abc import ABC
import time
import random
import numpy.random as nprandom
import py_trees

# Leaerboard Import
import sys, os
# Add paths to get leaderboard to work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../leaderboard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../scenario_runner'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './'))
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from leaderboard.scenarios.route_scenario import scenario_sampling, build_scenario_instances, convert_transform_to_location, Trigger
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# print(leaderboard, leaderboard.__file__)
# interpolate_trajectory = route_manipulation.interpolate_trajectory
# print(leaderboard.utils.route_manipulation.interpolate_trajectory)
# TODO make sure carla import works
import carla

#TODO add handling for offline map - needed for leaderboard


class Carla910Interface():

    def __init__(self, config, log_dir):
        self.config = config

        # Instantiate and start server
        self.server = CarlaServer(config)

        self.client = None

        self.log_dir = log_dir

        self.setup()

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        self.client = self._spawn_client()

        # Get the world
        self.world = self.client.load_world(self.config.scenario_config.city_name)
        # self.world = self.client.get_world()

        # Temporary
        self.spectator = self.world.get_spectator()


        # Update the settings from the config
        settings = self.world.get_settings()
        if(self.config.sync_mode):
            settings.synchronous_mode = True
        if self.config.server_fps is not None and self.config.server_fps != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config.server_fps)

        # Enable rendering
        # TODO render according to config settings
        settings.no_rendering_mode = False

        self.world.apply_settings(settings)

        # Sleep to allow for settings to update
        time.sleep(5)

        # Retrieve map
        self.map = self.world.get_map()

        # Get blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Instantiate a vehicle manager to handle other actors
        self.actor_fleet = ActorManager910(self.config, self.client, self.log_dir)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        self.scenario_index = 0

        print("server_version", self.client.get_server_version())

    def _spawn_client(self, hostname='localhost', port_number=None):
        port_number = self.server.server_port
        client = carla.Client(hostname, port_number)
        client.set_timeout(self.config.client_timeout_seconds)

        return client

    def _set_updated_scenario(self, unseen=False, town="Town01", index=0):
        if self.config.scenario_config.scenarios == "straight":
            source_idx, destination_idx = scenarios.get_straight_path_updated(unseen, town, index)
        elif self.config.scenario_config.scenarios == "curved":
            source_idx, destination_idx = scenarios.get_curved_path_updated(unseen, town, index)
        elif self.config.scenario_config.scenarios == "navigation" or self.config.scenario_config.scenarios == "dynamic_navigation":
            source_idx, destination_idx = scenarios.get_navigation_path_updated(unseen, town, index)
        else:
            raise ValueError("Scenarios Config not set!")

        self.source_transform = self.spawn_points[source_idx]
        self.destination_transform = self.spawn_points[destination_idx]

    def _set_scenario(self, unseen=False, town="Town01", index=0):
        if self.config.scenario_config.scenarios == "straight":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_straight_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_straight_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "long_straight":
            self.source_transform, self.destination_transform = scenarios.get_long_straight_path(unseen, town)
        elif self.config.scenario_config.scenarios == "long_straight_junction":
            self.source_transform, self.destination_transform = scenarios.get_long_straight_junction_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "straight_dynamic":
            self.source_transform, self.destination_transform = scenarios.get_straight_dynamic_path(unseen, town)
            self.source_transform, self.destination_transform = scenarios.get_crowded_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "straight_crowded":
            self.source_transform, self.destination_transform = scenarios.get_straight_crowded_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "town3":
            self.source_transform, self.destination_transform = scenarios.get_curved_town03_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "left_right_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_right_randomly(unseen)
        elif self.config.scenario_config.scenarios == "right_curved":
            self.source_transform, self.destination_transform = scenarios.get_right_turn(unseen)
        elif self.config.scenario_config.scenarios == "left_curved":
            self.source_transform, self.destination_transform = scenarios.get_left_turn(unseen)
        elif self.config.scenario_config.scenarios == "t_junction":
            self.source_transform, self.destination_transform = scenarios.get_t_junction_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "curved":
            # self.source_transform, self.destination_transform = scenarios.get_fixed_long_curved_path_Town01()
            self.source_transform, self.destination_transform = scenarios.get_curved_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "navigation" or self.config.scenario_config.scenarios == "dynamic_navigation":
            self.source_transform, self.destination_transform = scenarios.get_navigation_path(unseen, town, index)
        elif self.config.scenario_config.scenarios == "no_crash_empty" or self.config.scenario_config.scenarios == "no_crash_regular" or self.config.scenario_config.scenarios == "no_crash_dense":
            source_idx, destination_idx = scenarios.get_no_crash_path(unseen, town, index)
            self.source_transform = self.spawn_points[source_idx]
            self.destination_transform = self.spawn_points[destination_idx]
        else:
            raise ValueError("Invalid Scenario Type {}. Check scenario config!".format(self.config.scenario_config.scenarios))

    def reset(self, unseen = False, index = 0):
        ### Delete old actors
        self.actor_fleet.destroy_actors()

        if self.config.scenario_config.scenarios in ["long_straight", "long_straight_junction"] and not unseen:
            # Way to test two scenarios with and without dynamic actors
            # in training run in long_straight scenario
            self.scenario_index = (self.scenario_index + 1) % self.config.scenario_config.num_episodes
        else:
            self.scenario_index = index

        ## Set the new scenarios
        if self.config.scenario_config.use_scenarios and (self.config.scenario_config.city_name == "Town01" or self.config.scenario_config.city_name == "Town02"):
            if self.config.scenario_config.updated_scenarios:
                self._set_updated_scenario(unseen=unseen, index=self.scenario_index, town=self.config.scenario_config.city_name)
            else:
                self._set_scenario(unseen=unseen, index=self.scenario_index, town=self.config.scenario_config.city_name)
        else:
            self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)

        ### Spawn new actors
        self.actor_fleet.spawn(self.source_transform, unseen)

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            transform = self.actor_fleet.get_ego_vehicle_transform()
            self.spectator.set_transform(transform)
            world_frame = self.world.tick()


        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(transform)


        # Create a global planner to generate dense waypoints along route
        self.global_planner = planner.GlobalPlanner()

        ### Setup the global planner
        self.dense_waypoints  = self.global_planner.trace_route(self.map,
                                self.source_transform, self.destination_transform)

        self.global_planner.set_global_plan(self.dense_waypoints)

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.next_cmds = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)


        ep_measurements = {
            'next_orientation' : next_orientation,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'next_waypoints' : self.next_waypoints,
            'next_cmds': self.next_cmds,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : ego_vehicle_transform,
            'ego_vehicle_velocity' : ego_vehicle_velocity
        }

        control = {
            "target_speed" : 0.0,
            "control_steer" : 0.0,
            "control_throttle" : 0.0,
            "control_brake" : 0.0,
            "control_reverse" : False,
            "control_hand_brake" : False
        }

        # Create a copy of sensor_readings and ep_measurements to return
        obs = {**sensor_readings, **ep_measurements, **control}
        return obs


    def step(self, action):
        control = self.actor_fleet.step(action)

        world_frame = self.world.tick()

        self.actor_fleet.check_for_vehicle_elimination()

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)
        location = self.actor_fleet.ego_vehicle._vehicle.get_location()

        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(transform)

        sensor_readings["location"] = location

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.next_cmds = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        ep_measurements = {
            'next_orientation' : next_orientation,
            "next_waypoints" : self.next_waypoints,
            'next_cmds': self.next_cmds,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : ego_vehicle_transform,
            'ego_vehicle_velocity' : ego_vehicle_velocity,
            'location' : location
        }

        obs = {**control, **ep_measurements, **sensor_readings}

        return obs


    def get_actor_list(self):
        return self.actor_fleet.actor_list

    def get_ego_vehicle(self):
        return self.actor_fleet.ego_vehicle

    def get_traffic_actors(self):
        return self.traffic_actors

    def get_map(self):
        return self.map


    def close(self):
        self.actor_fleet.destroy_actors()

        if self.server is not None:
            self.server.close()



class Carla910Interface_Leaderboard:
    def __init__(self, config, log_dir):
        self.config = config

        # Instantiate and start server
        # print(23)
        self.server = CarlaServer(config)

        self.client = None

        self.log_dir = log_dir

        time.sleep(10)
        self.setup()

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        self.client = self._spawn_client()
        print(self.client.get_available_maps())
        self.avail_map = {name[-6:]: name for name in self.client.get_available_maps()}
        self._set_world_and_map(self.config.scenario_config.city_name)
        print("server_version", self.client.get_server_version())
        # print(os.getcwd())
        self.world_annotations = RouteParser.parse_annotations_file(
            '../../leaderboard/data/all_towns_traffic_scenarios_public.json')
        # print(self.world_annotations)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(4050)


    def _set_world_and_map(self, town_name):
        # Get the world
        self.curr_town = town_name
        self.world = self.client.load_world(self.curr_town)
        # Temporary
        self.spectator = self.world.get_spectator()


        # Update the settings from the config
        settings = self.world.get_settings()
        if(self.config.sync_mode):
            settings.synchronous_mode = True
        if self.config.server_fps is not None and self.config.server_fps != 0:
            settings.fixed_delta_seconds =  1.0 / float(self.config.server_fps)

        # Enable rendering
        settings.no_rendering_mode = False

        self.world.apply_settings(settings)

        # Sleep to allow for settings to update
        time.sleep(5)

        # Retrieve map
        self.map = self.world.get_map()

        # Get blueprints
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        # Instantiate a vehicle manager to handle other actors
        self.actor_fleet = ActorManager910_Leaderboard(self.config, self.client, self.log_dir)

        # Get traffic lights
        self.traffic_actors = self.world.get_actors().filter("*traffic_light*")

        self.scenario_index = 0

        CarlaDataProvider.set_world(self.world)


    def _spawn_client(self, hostname='localhost', port_number=None):
        #TODO switch back to getting port from server
        port_number = self.server.server_port
        # port_number = 2000
        client = carla.Client(hostname, port_number)
        # print(self.config)
        client.set_timeout(self.config.client_timeout_seconds)

        return client


    def _set_scenario(self, unseen=False, town="Town01", index=0):
        _upd_town = town

        if self.config.scenario_config.scenarios == "challenge_train_scenario":
            self.source_transform, self.destination_transform, self.wps_list, _upd_town = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=self.config.scenario_config.min_num_eps_before_switch_town,
                avail_map_list=self.avail_map.keys(), mode='train')
        elif self.config.scenario_config.scenarios == "challenge_test_scenario":
            self.source_transform, self.destination_transform, self.wps_list, _upd_town  = scenarios.get_leaderboard_route(
                unseen, curr_town=self.curr_town, index=index, max_idx=self.config.scenario_config.min_num_eps_before_switch_town,
                avail_map_list=self.avail_map.keys(), mode='test')
        else:
            raise ValueError("Scenarios Config not set!")

        if _upd_town != town: # switch to a new town
            self._set_world_and_map(_upd_town)
            # self.reset()



    def check_subset(self, pt):
        for spawn_pt in self.spawn_points:
            if pt.location.distance(spawn_pt.location)<1:
                return True
        return False

    def reset(self, unseen=False, index=0):
        ### Delete old actors
        self.actor_fleet.destroy_actors()

        ## Set the new scenarios
        # if self.config.scenario_config.use_scenarios:
        self._set_scenario(unseen=unseen, index=self.scenario_index, town=self.curr_town)
        self.scenario_index += 1

        ### Spawn new actors
        self.actor_fleet.spawn(self.source_transform, unseen)

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            transform = self.actor_fleet.get_ego_vehicle_transform()
            # self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
            world_frame = self.world.tick()


        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        # Create a global planner to generate dense waypoints along route
        self.global_planner = planner.GlobalPlanner()

        ### Setup the global planner
        if 'challenge' in self.config.scenario_config.scenarios:
            # print(213, len(self.wps_list))
            _, self.route, self._global_plan_world_coord = interpolate_trajectory(self.world, self.wps_list)
            CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))
            # print(222, len(self._global_plan_world_coord), self._global_plan_world_coord[0])
            self.dense_waypoints = self._global_plan_world_coord
        else:
            self.dense_waypoints  = self.global_planner.trace_route(self.map,
                                    self.source_transform, self.destination_transform)
            # print(self.dense_waypoints)

        self.global_planner.set_global_plan(self.dense_waypoints)

        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(
            self.curr_town, self.route, self.world_annotations)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = scenario_sampling(potential_scenarios_definitions)
        # print(236, self.sampled_scenarios_definitions)
        self.scenarios = build_scenario_instances(self.world, self.actor_fleet.vehicle_actor, self.sampled_scenarios_definitions, debug_mode=1)
        # print(244, self.scenarios)
        self.running = Trigger(self.world, self.actor_fleet.vehicle_actor, self.route, self.scenarios, debug_mode=1)

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.next_cmds = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)


        ep_measurements = {
            'next_orientation' : next_orientation,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'next_waypoints' : self.next_waypoints,
            'next_cmds': self.next_cmds,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : ego_vehicle_transform,
            'ego_vehicle_velocity' : ego_vehicle_velocity
        }

        control = {
            "target_speed" : 0.0,
            "control_steer" : 0.0,
            "control_throttle" : 0.0,
            "control_brake" : 0.0,
            "control_reverse" : False,
            "control_hand_brake" : False
        }

        # Create a copy of sensor_readings and ep_measurements to return
        obs = {**sensor_readings, **ep_measurements, **control}
        return obs


    def step(self, action):
        control = self.actor_fleet.step(action)

        CarlaDataProvider.on_carla_tick()
        world_frame = self.world.tick()
        # print(290, self.running.scenario.scenario_tree.status)
        self.running.scenario.scenario_tree.tick_once()
        # print(291, self.running.scenario.scenario_tree.status)
        # if self.running.debug_mode == 1:

        # print("\n")
        # py_trees.display.print_ascii_tree(
        #     self.running.scenario.scenario_tree, show_status=True)
        # sys.stdout.flush()
        # print("\n\n\n\n\n")

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)
        location = self.actor_fleet.ego_vehicle._vehicle.get_location()

        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        sensor_readings["location"] = location

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.next_cmds = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        ep_measurements = {
            'next_orientation' : next_orientation,
            "next_waypoints" : self.next_waypoints,
            'next_cmds': self.next_cmds,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : ego_vehicle_transform,
            'ego_vehicle_velocity' : ego_vehicle_velocity,
            'location' : location
        }

        obs = {**control, **ep_measurements, **sensor_readings}

        return obs


    def get_actor_list(self):
        return self.actor_fleet.actor_list

    def get_ego_vehicle(self):
        return self.actor_fleet.ego_vehicle

    def get_traffic_actors(self):
        return self.traffic_actors

    def get_map(self):
        return self.map

    def destroy_all_actors(self):
        # raise NotImplementedError()
        pass

    def close(self):
        self.actor_fleet.destroy_actors()

        if self.server is not None:
            self.server.close()
