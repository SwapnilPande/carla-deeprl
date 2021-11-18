import environment.carla_interfaces.scenarios_910 as scenarios
from environment.carla_interfaces.server import CarlaServer
from environment.carla_interfaces import planner
from environment.carla_interfaces.actor_manager import ActorManager910, ActorManager910_Leaderboard
from environment import env_util as util
from abc import ABC
import time
import random
import numpy as np
import numpy.random as nprandom


# print(leaderboard, leaderboard.__file__)
# interpolate_trajectory = route_manipulation.interpolate_trajectory
# print(leaderboard.utils.route_manipulation.interpolate_trajectory)
import carla

#TODO add handling for offline map - needed for leaderboard


class Carla910Interface():

    def __init__(self, config, log_dir, logger = None):
        self.config = config

        self.logger = logger

        # Instantiate and start server
        self.server = CarlaServer(config, logger = logger)

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
        settings.no_rendering_mode = not self.config.render_server

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
        # Initialize state for traffic lights
        self.traffic_light_state = {
            'initial_dist_to_red_light' : -1
        }

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
            while True:
                self.source_transform, self.destination_transform = random.choice(self.spawn_points), random.choice(self.spawn_points)
                distance = np.linalg.norm([
                    self.source_transform.location.x - self.destination_transform.location.x,
                    self.source_transform.location.y - self.destination_transform.location.y,
                    self.source_transform.location.z - self.destination_transform.location.z])
                if distance > 100:
                    break

        ### Spawn new actors
        self.actor_fleet.spawn(self.source_transform, unseen)

        # Create a global planner to generate dense waypoints along route
        self.global_planner = planner.GlobalPlanner()

        ### Setup the global planner
        self.dense_waypoints  = self.global_planner.trace_route(self.map,
                                self.source_transform, self.destination_transform)

        self.global_planner.set_global_plan(self.dense_waypoints)

        # If waypoint is behind vehicle, rotate the vehicle
        if abs(self.global_planner.get_next_orientation_new(self.source_transform)[0]) > .5:
            self.reset(unseen=unseen, index=index)

        # Tick for 15 frames to handle car initialization in air
        for _ in range(15):
            transform = self.actor_fleet.get_ego_vehicle_transform()
            self.spectator.set_transform(transform)
            world_frame = self.world.tick()

        left_steer = self.actor_fleet.ego_vehicle._vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        right_steer = self.actor_fleet.ego_vehicle._vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        # Average steering angle between front two wheels, and normalize by dividing by 90
        steer_angle = (left_steer + right_steer) / (2* 90)

        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(transform)

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()
        speed = util.get_speed_from_velocity(ego_vehicle_velocity)

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.all_waypoints = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)


        ep_measurements = {
            'next_orientation' : next_orientation,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'next_waypoints' : self.next_waypoints,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : ego_vehicle_transform,
            'ego_vehicle_velocity' : ego_vehicle_velocity,
            'waypoints' : self.all_waypoints,
            "steer_angle" : steer_angle,
            'speed' : speed,
            'autopilot_action' : self.get_autopilot_action()
        }

        control = {
            "target_speed" : 0.0,
            "control_steer" : 0.0,
            "control_throttle" : 0.0,
            "control_brake" : 0.0,
            "control_reverse" : False,
            "control_hand_brake" : False
        }

        ep_measurements["traffic_light"] = self.get_traffic_light_states()
        ep_measurements["obstacles"] = self.get_obstacle_states(self.next_waypoints)

        # Create a copy of sensor_readings and ep_measurements to return
        obs = {**sensor_readings, **ep_measurements, **control}
        return obs


    def step(self, action):
        control = self.actor_fleet.step(action)

        world_frame = self.world.tick()

        self.actor_fleet.check_for_vehicle_elimination()

        sensor_readings = self.actor_fleet.sensor_manager.get_sensor_readings(world_frame)
        location = self.actor_fleet.ego_vehicle._vehicle.get_location()

        left_steer = self.actor_fleet.ego_vehicle._vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        right_steer = self.actor_fleet.ego_vehicle._vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        # Average steering angle between front two wheels, and normalize by dividing by 90
        steer_angle = (left_steer + right_steer) / (2* 90)

        transform = self.actor_fleet.get_ego_vehicle_transform()
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))

        sensor_readings["location"] = location

        ego_vehicle_transform = self.actor_fleet.get_ego_vehicle_transform()
        ego_vehicle_velocity = self.actor_fleet.get_ego_vehicle_velocity()
        speed = util.get_speed_from_velocity(ego_vehicle_velocity)

        next_orientation, \
        self.dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        self.next_wp_angles, \
        self.next_wp_vectors, \
        self.all_waypoints = self.global_planner.get_next_orientation_new(ego_vehicle_transform)

        ep_measurements = {
            'next_orientation' : next_orientation,
            "next_waypoints" : self.next_waypoints,
            'distance_to_goal_trajec' : self.dist_to_trajectory,
            'dist_to_trajectory' : self.dist_to_trajectory,
            'dist_to_goal' : ego_vehicle_transform.location.distance(self.destination_transform.location),
            'ego_vehicle_location' : [ego_vehicle_transform.location.x, ego_vehicle_transform.location.y, ego_vehicle_transform.rotation.yaw],
            'ego_vehicle_velocity' : ego_vehicle_velocity,
            'location' : location,
            'waypoints' : self.all_waypoints,
            "steer_angle" : steer_angle,
            "speed" : speed,
            'autopilot_action' : self.get_autopilot_action()
        }

        ep_measurements["traffic_light"] = self.get_traffic_light_states()
        ep_measurements["obstacles"] = self.get_obstacle_states(self.next_waypoints)

        obs = {**control, **ep_measurements, **sensor_readings}

        return obs

    def get_traffic_light_states(self):
        ego_vehicle = self.actor_fleet.ego_vehicle
        traffic_actor, dist, traffic_light_orientation, nearest_light_transform = ego_vehicle.find_nearest_traffic_light(self.traffic_actors)


        if traffic_light_orientation is not None:
            self.traffic_light_state['traffic_light_orientation'] = traffic_light_orientation
        else:
            self.traffic_light_state['traffic_light_orientation'] = -1

        if traffic_actor is not None:
            if traffic_actor.state == carla.TrafficLightState.Red:
                self.traffic_light_state['red_light_dist'] = dist

                if self.traffic_light_state['initial_dist_to_red_light'] == -1 \
                    or (self.traffic_light_state['nearest_traffic_actor_id'] != -1 and traffic_actor.id != self.traffic_light_state['nearest_traffic_actor_id']):
                    self.traffic_light_state['initial_dist_to_red_light'] = dist

            else:
                self.traffic_light_state['red_light_dist'] = -1
                self.traffic_light_state['initial_dist_to_red_light'] = -1

            self.traffic_light_state['nearest_traffic_actor_id'] = traffic_actor.id
            self.traffic_light_state['nearest_traffic_actor_state'] = str(traffic_actor.state)
        else:
            self.traffic_light_state['red_light_dist'] = -1
            self.traffic_light_state['initial_dist_to_red_light'] = -1
            self.traffic_light_state['nearest_traffic_actor_id'] = -1
            self.traffic_light_state['nearest_traffic_actor_state'] = None

        self.traffic_light_state['dist_to_light'] = dist
        self.traffic_light_state['nearest_traffic_actor_location'] = nearest_light_transform
        return self.traffic_light_state


    def get_obstacle_states(self, next_waypoints):
        obstacle_state = {}
        obstacle_state['obstacle_visible'] = False
        obstacle_state['obstacle_orientation'] = -1

        min_obs_distance = 100000000
        found_obstacle = False

        ego_vehicle_actor = self.actor_fleet.ego_vehicle._vehicle
        map = self.map

        # try:
        for target_vehicle in self.world.get_actors():
            # do not account for the ego vehicle
            if target_vehicle.id == ego_vehicle_actor.id or "vehicle" not in target_vehicle.type_id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = map.get_waypoint(target_vehicle.get_location())
            d_bool, d_angle, distance = util.is_within_distance_ahead(target_vehicle.get_transform(),
                                        ego_vehicle_actor.get_transform(),
                                        self.config.obs_config.vehicle_proximity_threshold)

            if not d_bool:
                continue
            else:
                if not util.check_if_vehicle_in_same_lane(ego_vehicle_actor, target_vehicle, next_waypoints, map):
                    continue

                found_obstacle = True
                obstacle_state['obstacle_visible'] = True
                obstacle_state['obstacle_orientation'] = d_angle

                if distance < min_obs_distance:
                    obstacle_state['obstacle_dist'] = distance
                    obstacle_state['obstacle_speed'] = util.get_speed_from_velocity(target_vehicle.get_velocity())

                    min_obs_distance = distance
        # except:
        #     import ipdb; ipdb.set_trace()

        if not found_obstacle:
            obstacle_state['obstacle_dist'] = -1
            obstacle_state['obstacle_speed'] = -1



        return obstacle_state

    def get_autopilot_action(self, target_speed = 1.0):
        hazard_detected = self.get_ego_vehicle().check_for_hazard()

        if hazard_detected:
            return np.array([0,-1])
        else:
            waypoint = self.next_waypoints[0]
            steer = self.actor_fleet.lateral_controller.pid_control(waypoint)
            steer = np.clip(steer, -1, 1)
            return np.array([steer, target_speed])


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
