import numpy as np
import os
import random
import time

# Need to change the imports to contain env flag
import environment.carla_interfaces.scenarios_910 as scenarios
from environment.carla_interfaces.agents.navigation.agent import Agent
from environment.carla_interfaces.agents.navigation.basic_agent import BasicAgent
import environment.carla_interfaces.sensors as sensors
import environment.carla_interfaces.controller as controller
from environment.config.config import DISCRETE_ACTIONS
from environment import env_util as util

# Need to change the imports to contain env flag
import carla
from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation

# Leaerboard Import
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../leaderboard'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../scenario_runner'))
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

class ActorManager910():
    def __init__(self, config, client, log_dir):
        '''
        Manages ego vehicle, other actors and sensors
        Assumes that sensormanager is always attached to ego vehicle

        Common/High level attributes are:
        1) Spawn points (Used for spwaning actors and also by planner)
        2) Blueprints
        '''
        self.config = config
        self.world = client.get_world()


        ################################################
        # Spawn points
        ################################################
        self.spawn_points = self.world.get_map().get_spawn_points()
        # Only randomize order of spawn points if testing
        if self.config.testing:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config.spawn_points_fixed_idx]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            # np.save(os.path.join(log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        ################################################
        # Blueprints
        ################################################
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config.scenario_config.disable_two_wheeler:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]


        # tm is valid for carla0.9.10. If using carla0.9.6, this has to be commented out
        # This is for autopilot purpose on npcs
        # push it to spawn_npc() function?
        tm_port = np.random.randint(10000, 60000)
        self.tm = client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)

        self.actor_list = []

    def spawn(self, source_transform, unseen):

        # Parameters for ego vehicle
        self.ego_vehicle = self.spawn_ego_vehicle(source_transform)
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.args_lateral_dict = {
            'K_P': 0.88,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1/10.0}
        self.controller = controller.PIDLongitudinalController(K_P=self.args_longitudinal_dict['K_P'], K_D=self.args_longitudinal_dict['K_D'], K_I=self.args_longitudinal_dict['K_I'], dt=self.args_longitudinal_dict['dt'])
        self.lateral_controller = controller.PIDLateralController(self.vehicle_actor, K_P=self.args_lateral_dict['K_P'], K_D=self.args_lateral_dict['K_D'], K_I=self.args_lateral_dict['K_I'], dt=self.args_lateral_dict['dt'])
        self.target_speed = self.config.action_config.target_speed

        self.sensor_manager = self.spawn_sensors()
        # Check how to obtain the function argument value of 'unseen' variable
        if self.config.scenario_config.sample_npc:
            number_of_vehicles = np.random.randint(low=self.config.scenario_config.num_npc_lower_threshold,
                                                    high=self.config.scenario_config.num_npc_upper_threshold)
        else:
            number_of_vehicles = self.config.scenario_config.num_npc

        self.spawn_npc(number_of_vehicles, unseen)

    def spawn_ego_vehicle(self, source_transform):
        '''
        Spawns and return ego vehicle/Agent
        '''
        # Spawn the actor
        # Create an Agent object with that actor
        # Return the agent instance
        vehicle_bp = self.blueprint_library.find(self.config.scenario_config.vehicle_type)


        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        NUM_RETRIES = 10

        for _ in range(NUM_RETRIES):
            self.vehicle_actor = self.world.try_spawn_actor(vehicle_bp, source_transform)
            if self.vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(source_transform.location.x, source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_actors()              # Do we need this as ego vehicle is the first one to be spawned?
                time.sleep(60)

        if self.vehicle_actor is not None:
            self.actor_list.append(self.vehicle_actor)
            # Need to move this variable to carla interface file
            # self.location = self.vehicle_actor.get_location()
        else:
            raise Exception("Failed in spawning vehicle actor.")

        # Agent uses proximity_threshold to detect traffic lights.
        # Hence we use traffic_light_proximity_threshold while creating an Agent.
        vehicle_agent = BasicAgent(self.vehicle_actor, self.config.obs_config.traffic_light_proximity_threshold)
        return vehicle_agent

    def get_ego_vehicle_transform(self):
        return self.ego_vehicle._vehicle.get_transform()

    def get_ego_vehicle_velocity(self):
        return self.ego_vehicle._vehicle.get_velocity()

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """

        episode_measurements = {}

        if self.config.action_config.action_type != "control":
            action = action.flatten()

        if self.config.action_config.action_type is "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])
        elif self.config.action_config.action_type is "merged_gas":
            steer = float(action[0])
            gas = float(action[1])
            # gas = gas + 0.25
            gas = np.clip(gas, 0.0, 0.7)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "steer_only":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(20.0)
            current_speed = util.get_speed_from_velocity(self.ego_vehicle.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "merged_speed_scaled_tanh":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = (action[1] * 1.5) + 1
            target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "merged_speed_pid_test":
            # steer = float(action[0])
            steer = (float(action[0]))
            target_speed = float(action[1])
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "discrete":
            # Discrete actions
            # No need to clip actions in case of discrete state-space
            # since it is chosen to be in range.
            discrete_actions = DISCRETE_ACTIONS[int(action)]
            target_speed, steer = float(discrete_actions[0]), float(discrete_actions[1])
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "control":
            target_speed = -1
            episode_measurements["target_speed"] = target_speed
            return action, episode_measurements
        else:
            raise Exception("Invalid Action Type")

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        episode_measurements["target_speed"] = target_speed

        episode_measurements['control_steer'] = control.steer
        episode_measurements['control_throttle'] = control.throttle
        episode_measurements['control_brake'] = control.brake
        episode_measurements['control_reverse'] = control.reverse
        episode_measurements['control_hand_brake'] = control.hand_brake

        return control, episode_measurements

    def step(self, action):
        control, ep_measurements = self.get_control(action)
        self.ego_vehicle._vehicle.apply_control(control)

        return ep_measurements

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        self.actor_list = [actor for actor in self.actor_list if actor.is_alive]

    def spawn_sensors(self):
        if self.ego_vehicle is None:
            print("Not spwaning sensors as the parent actor is not initialized properly")
            return None
        sensor_manager = sensors.SensorManager(self.config, self.ego_vehicle._vehicle)
        sensor_manager.spawn()
        for k,v in sensor_manager.sensors.items():
            self.actor_list.append(v.sensor)
        return sensor_manager

    def spawn_npc(self, number_of_vehicles, unseen):
        npc_spawn_points = self.pick_npc_spawn_points(number_of_vehicles, unseen)
        count = number_of_vehicles
        for spawn_point in npc_spawn_points:
            #$
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def pick_npc_spawn_points(self, number_of_vehicles, unseen):
        if self.config.scenario_config.scenarios == "straight_dynamic":
            # vehicle spawn_points corresponding to 84, 40
            spawn_points = [Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
                        Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016))]

            # vehicle spawn_points corresponding to 96, 140
            # spawn_points = [Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
            # Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))]
        elif self.config.scenario_config.scenarios == "crowded":
            spawn_points = scenarios.get_crowded_npcs(number_of_vehicles)
            print('CROWDED SPAWNING: ', spawn_points)
        elif self.config.scenario_config.scenarios in ["long_straight", "long_straight_junction"]:
            spawn_points_1 = scenarios.get_long_straight_npcs()
            if unseen:
                if self.config.test_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config.train_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)

        elif self.config.scenario_config.scenarios == "straight_crowded":
            spawn_points = scenarios.get_straight_crowded_npcs(number_of_vehicles)
            print('STRAIGHT CROWDED SPAWNING: ', spawn_points)
        elif self.config.scenario_config.scenarios == "town3":
            spawn_points = scenarios.get_curved_town03_npcs(number_of_vehicles)
            print('TOWN 3 SPAWNING: ', spawn_points)

        else:
            # Testing
            if unseen:
                if self.config.test_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config.train_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)


        if self.config.verbose:
            print('found %d spawn points.' % len(spawn_points))

        return spawn_points

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # TODO: uncomment below to enable autopilot
        if not self.config.scenario_config.scenarios == "straight_dynamic":
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()
        if vehicle is not None:
            self.actor_list.append(vehicle)
            if not self.config.scenario_config.scenarios == "straight_dynamic":
                vehicle.set_autopilot(True, tm_port)

            if self.config.verbose:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False

    def destroy_actors(self):
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))


class ActorManager910_Leaderboard():
    def __init__(self, config, client, log_dir):
        '''
        Manages ego vehicle, other actors and sensors
        Assumes that sensormanager is always attached to ego vehicle

        Common/High level attributes are:
        1) Spawn points (Used for spwaning actors and also by planner)
        2) Blueprints
        '''
        self.config = config
        self.world = client.get_world()


        ################################################
        # Spawn points
        ################################################
        self.spawn_points = self.world.get_map().get_spawn_points()
        # Only randomize order of spawn points if testing
        if self.config.testing:
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in self.config.spawn_points_fixed_idx]
        else:
            spawn_pt_idx = np.random.permutation(len(self.spawn_points))
            np.save(os.path.join(log_dir, "spawn_pt_order"), spawn_pt_idx)
            self.spawn_points_fixed_order =  [self.spawn_points[i] for i in spawn_pt_idx]

        ################################################
        # Blueprints
        ################################################
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        if self.config.scenario_config.disable_two_wheeler:
            self.vehicle_blueprints = [x for x in self.vehicle_blueprints if int(x.get_attribute('number_of_wheels')) == 4]


        # tm is valid for carla0.9.10. If using carla0.9.6, this has to be commented out
        # This is for autopilot purpose on npcs
        # push it to spawn_npc() function?

        tm_port = np.random.randint(10000, 60000)
        self.tm = client.get_trafficmanager(tm_port)
        self.tm.set_synchronous_mode(True)

        self.actor_list = []

    def spawn(self, source_transform, unseen):
        # Parameters for ego vehicle
        self.ego_vehicle = self.spawn_ego_vehicle(source_transform)
        self.args_longitudinal_dict = {
            'K_P': 0.1,
            'K_D': 0.0005,
            'K_I': 0.4,
            'dt': 1/10.0}
        self.args_lateral_dict = {
            'K_P': 0.88,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1/10.0}
        self.controller = controller.PIDLongitudinalController(K_P=self.args_longitudinal_dict['K_P'], K_D=self.args_longitudinal_dict['K_D'], K_I=self.args_longitudinal_dict['K_I'], dt=self.args_longitudinal_dict['dt'])
        self.lateral_controller = controller.PIDLateralController(self.vehicle_actor, K_P=self.args_lateral_dict['K_P'], K_D=self.args_lateral_dict['K_D'], K_I=self.args_lateral_dict['K_I'], dt=self.args_lateral_dict['dt'])
        self.target_speed = self.config.action_config.target_speed

        self.sensor_manager = self.spawn_sensors()
        # Check how to obtain the function argument value of 'unseen' variable
        if self.config.scenario_config.sample_npc:
            number_of_vehicles = np.random.randint(low=self.config.scenario_config.num_npc_lower_threshold, high=self.config.scenario_config.num_npc_upper_threshold)
        else:
            number_of_vehicles = self.config.scenario_config.num_npc

        self.spawn_npc(number_of_vehicles, unseen)

    def spawn_ego_vehicle(self, source_transform):
        '''
        Spawns and return ego vehicle/Agent
        '''
        # Spawn the actor
        # Create an Agent object with that actor
        # Return the agent instance
        #try:
        vehicle_bp = self.blueprint_library.find(self.config.scenario_config.vehicle_type)
        # vehicle_bp = self.blueprint_library.find(random.choice(self.config['vehicle_types']))
        #except Exception as e:
        #    print("Error during vehicle creation: {}".format(traceback.format_exc()))


        # Spawning vehicle actor with retry logic as it fails to spawn sometimes
        NUM_RETRIES = 5

        #TODO Do we want this to be vehicle actor?
        for _ in range(NUM_RETRIES):
            try:
                # Need to check about passing source_transform
                self.vehicle_actor = CarlaDataProvider.request_new_actor(self.config.scenario_config.vehicle_type,
                                                                        source_transform,
                                                                        'hero',)
            except:
                self.vehicle_actor = None
            # self.vehicle_actor = self.world.try_spawn_actor(vehicle_bp, source_transform)
            if self.vehicle_actor is not None:
                break
            else:
                print("Unable to spawn vehicle actor at {0}, {1}.".format(source_transform.location.x, source_transform.location.y))
                print("Number of existing actors, {0}".format(len(self.actor_list)))
                self.destroy_actors()              # Do we need this as ego vehicle is the first one to be spawned?
                # time.sleep(120)

        if self.vehicle_actor is not None:
            self.actor_list.append(self.vehicle_actor)
            # Need to move this variable to carla interface file
            # self.location = self.vehicle_actor.get_location()
        else:
            raise Exception("Failed in spawning vehicle actor.")

        # Agent uses proximity_threshold to detect traffic lights.
        # Hence we use traffic_light_proximity_threshold while creating an Agent.
        vehicle_agent = BasicAgent(self.vehicle_actor, self.config.obs_config.traffic_light_proximity_threshold)
        return vehicle_agent

    def get_ego_vehicle_transform(self):
        return self.ego_vehicle._vehicle.get_transform()

    def get_ego_vehicle_velocity(self):
        return self.ego_vehicle._vehicle.get_velocity()

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """

        episode_measurements = {}

        if self.config.action_config.action_type != "control":
            action = action.flatten()

        if self.config.action_config.action_type is "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])
        elif self.config.action_config.action_type is "merged_gas":
            steer = float(action[0])
            gas = float(action[1])
            # gas = gas + 0.25
            gas = np.clip(gas, 0.0, 0.7)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "steer_only":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(20.0)
            current_speed = util.get_speed_from_velocity(self.ego_vehicle.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "throttle_only":
            steer = float(0.0)
            target_speed = float(np.clip(action[0], 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "merged_speed":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip(action[1] + 10.0, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            throttle = self.controller.pid_control(target_speed, current_speed)
            brake = float(0.0)
        elif self.config.action_config.action_type == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "merged_speed_scaled_tanh":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = (action[1] * 1.5) + 1
            target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "merged_speed_pid_test":
            # steer = float(action[0])
            steer = (float(action[0]))
            target_speed = float(action[1])
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "discrete":
            # Discrete actions
            # No need to clip actions in case of discrete state-space
            # since it is chosen to be in range.
            discrete_actions = DISCRETE_ACTIONS[int(action)]
            target_speed, steer = float(discrete_actions[0]), float(discrete_actions[1])
            current_speed = util.get_speed_from_velocity(self.vehicle_actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.config.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.config.action_config.action_type == "control":
            target_speed = -1
            episode_measurements["target_speed"] = target_speed
            return action, episode_measurements

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)

        episode_measurements["target_speed"] = target_speed

        episode_measurements['control_steer'] = control.steer
        episode_measurements['control_throttle'] = control.throttle
        episode_measurements['control_brake'] = control.brake
        episode_measurements['control_reverse'] = control.reverse
        episode_measurements['control_hand_brake'] = control.hand_brake

        return control, episode_measurements

    def step(self, action):
        control, ep_measurements = self.get_control(action)
        self.ego_vehicle._vehicle.apply_control(control)

        return ep_measurements

    def check_for_vehicle_elimination(self):
        # https://github.com/carla-simulator/carla/issues/3860
        self.actor_list = [actor for actor in self.actor_list if actor.is_alive]

    def spawn_sensors(self):
        if self.ego_vehicle is None:
            print("Not spwaning sensors as the parent actor is not initialized properly")
            return None
        sensor_manager = sensors.SensorManager(self.config, self.ego_vehicle._vehicle)
        sensor_manager.spawn()
        for k,v in sensor_manager.sensors.items():
            self.actor_list.append(v.sensor)
        return sensor_manager

    def spawn_npc(self, number_of_vehicles, unseen):
        npc_spawn_points = self.pick_npc_spawn_points(number_of_vehicles, unseen)
        count = number_of_vehicles
        for spawn_point in npc_spawn_points:
            #$
            if self.try_spawn_random_vehicle_at(self.vehicle_blueprints, spawn_point):
                count -= 1
            if count <= 0:
                break

    def pick_npc_spawn_points(self, number_of_vehicles, unseen):
        if self.config.scenario_config.scenarios == "straight_dynamic":
            # vehicle spawn_points corresponding to 84, 40
            spawn_points = [Transform(Location(x=-2.4200193881988525, y=187.97000122070312, z=1.32), Rotation(yaw=89.9996109008789)),
                        Transform(Location(x=1.5599803924560547, y=187.9700164794922, z=1.32), Rotation(yaw=-90.00040435791016))]

            # vehicle spawn_points corresponding to 96, 140
            # spawn_points = [Transform(Location(x=88.61997985839844, y=249.42999267578125, z=1.32), Rotation(yaw=90.00004577636719)),
            # Transform(Location(x=92.10997772216797, y=249.42999267578125, z=1.32), Rotation(yaw=-90.00029754638672))]
        elif self.config.scenario_config.scenarios == "crowded":
            spawn_points = scenarios.get_crowded_npcs(number_of_vehicles)
            print('CROWDED SPAWNING: ', spawn_points)
        elif self.config.scenario_config.scenarios in ["long_straight", "long_straight_junction"]:
            spawn_points_1 = scenarios.get_long_straight_npcs()
            if unseen:
                if self.config.test_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config.train_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)

        elif self.config.scenario_config.scenarios == "straight_crowded":
            spawn_points = scenarios.get_straight_crowded_npcs(number_of_vehicles)
            print('STRAIGHT CROWDED SPAWNING: ', spawn_points)
        elif self.config.scenario_config.scenarios == "town3":
            spawn_points = scenarios.get_curved_town03_npcs(number_of_vehicles)
            print('TOWN 3 SPAWNING: ', spawn_points)

        else:
            # Testing
            if unseen:
                if self.config.test_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)
            else:
                if self.config.train_fixed_spawn_points:
                    spawn_points = self.spawn_points_fixed_order
                else:
                    spawn_points = self.spawn_points
                    random.shuffle(spawn_points)


        if self.config.verbose:
            print('found %d spawn points.' % len(spawn_points))

        return spawn_points

    def try_spawn_random_vehicle_at(self, blueprints, transform):
        # To spawn same type of vehicle
        blueprint = blueprints[0]
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        if not self.config.scenario_config.scenarios == "straight_dynamic":
            blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        tm_port = self.tm.get_port()
        if vehicle is not None:
            self.actor_list.append(vehicle)
            if not self.config.scenario_config.scenarios == "straight_dynamic":
                vehicle.set_autopilot(True, tm_port)

            if self.config.verbose:
                print('spawned %r at %s' % (vehicle.type_id, transform.location.x))
            return True
        return False

    def destroy_actors(self):
        for _ in range(len(self.actor_list)):
            try:
                actor = self.actor_list.pop()
                actor.destroy()
            except Exception as e:
                print("Error during destroying actor {0}:{1}: {2}".format(actor.type_id, actor.id,traceback.format_exc()))
