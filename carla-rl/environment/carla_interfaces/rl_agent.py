import numpy as np

import carla
from environment.carla_interfaces import planner
import environment.carla_interfaces.controller as controller
from environment import env_util as util
from copy import deepcopy

from leaderboard.autoagents.agent_wrapper import KillSimulator
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from environment.carla_interfaces.agents.navigation.basic_agent import BasicAgent
from environment.carla_interfaces.symbolic_utils import fetch_symbolic_dict

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


def get_entry_point():
    return 'RLAgent'


class FakeWaypoint:
    """
    This class is used to wrap the waypoints that leaderboard generates into a class that mimics carla waypoints
    We use this to match the global planner API
    """

    def __init__(self, transform):
        self.transform = transform

    @staticmethod
    def to_waypoint(waypoint_tuple):
        return FakeWaypoint(waypoint_tuple[0]), waypoint_tuple[1]


class RLAgent(AutonomousAgent):
    """
    Trained image agent
    """

    def setup(self, conf):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0

        # with open(path_to_conf_file, 'r') as f:
        #     config = yaml.safe_load(f)

        # for key, value in config.items():
        #     setattr(self, key, value)

        self.step = 0

        self.data_buffer = conf["data_buffer"]
        self.send_event = conf["send_event"]
        self.receive_event = conf["receive_event"]
        self.proximity_threshold = conf["proximity_threshold"]

        self.target_speed = conf["target_speed"]
        self.obs_config = conf["obs_config"]
        self.action_config = conf["action_config"]


    def initialize(self, input_data):
        """
        Initialization after scenario setup; runs once in the first run_step call
        """

        self.actor = CarlaDataProvider.get_hero_actor()
        self.basic_agent = BasicAgent(self.actor, self.proximity_threshold) # TODO: add proximity threshold to config
        self.world = self.actor.get_world()
        self.map = self.world.get_map()

        # Initialize dictionary for the current control input
        self.current_controls = {
            "target_speed" : 0.0,
            "control_steer" : 0.0,
            "control_throttle" : 0.0,
            "control_brake" : 0.0,
            "control_reverse" : False,
            "control_hand_brake" : False
        }


        self.global_planner = planner.GlobalPlanner()
        # Wrap the transforms for the waypoints into FakeWaypoints to match the API for the global planner
        # We don't use carla waypoints because they can't be instantiated from python, so we create our own
        # self.dense_waypoints = list(map(FakeWaypoint.to_waypoint, self._global_plan_world_coord))
        self.dense_waypoints = []
        for wp in self._global_plan_world_coord:
            self.dense_waypoints.append((self.map.get_waypoint(wp[0].location), wp[1]))

        # Get destination transform
        self.destination_transform = self.dense_waypoints[-1][0].transform
        self.global_planner.set_global_plan(self.dense_waypoints)
        # self.waypointer = Waypointer(self._global_plan, gps, use_ground_truth=True)
        # self.waypointer = Waypointer(self._global_plan_world_coord, ego_transform, dense=True)

        # Initialize sensor readings
        self.num_collisions = 0
        self.collision_at_last_step = False
        self.num_lane_invasions = 0
        self.lane_invasion_at_last_step = False

        # Define controllers
        # Parameters for ego vehicle
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
        self.lateral_controller = controller.PIDLateralController(self.actor, K_P=self.args_lateral_dict['K_P'], K_D=self.args_lateral_dict['K_D'], K_I=self.args_lateral_dict['K_I'], dt=self.args_lateral_dict['dt'])

        # Initialize state for traffic lights
        self.traffic_light_state = {
            'initial_dist_to_red_light' : -1
        }

    def sensors(self):
        #TODO move sensors to config
        sensors = [
            {'type': 'sensor.collision', 'id': 'collision_sensor'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': 0.0, 'id': 'GPS'},
            {'type': 'sensor.lane_invasion', 'id': 'lane_invasion_sensor'},
            # {'type': 'sensor.stitch_camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': 8, 'y': 0, 'z': 20, 'roll': 0.0, 'pitch': 270, 'yaw': 0.0,
            'width': 512, 'height': 512, 'fov': 90, 'id': "sensor.camera.rgb/top"},
        ]

        return sensors

    def get_sensor_readings(self, input_data):
        sensor_readings = {}

        # Increment count of collisions if crashed
        self.num_collisions += not self.collision_at_last_step and input_data["collision_sensor"][1]["collision"]
        self.collision_at_last_step = input_data["collision_sensor"][1]["collision"]

        self.num_lane_invasions = not self.lane_invasion_at_last_step and input_data["lane_invasion_sensor"][1]
        self.lane_invasion_at_last_step = input_data["lane_invasion_sensor"][1]

        actor_id = None
        actor_type = None
        if(input_data["collision_sensor"][1]["actor"] is not None):
            actor_id = input_data["collision_sensor"][1]["actor"].id
            actor_type = input_data["collision_sensor"][1]["actor"].type_id
        sensor_readings["collision_sensor"] = {
            'num_collisions' : self.num_collisions,
            "collision_actor_id" : actor_id,
            "collision_actor_type" : actor_type,
        }

        sensor_readings["lane_invasion_sensor"] = {
            'num_lane_intersections' : self.num_lane_invasions,
            "out_of_road" : input_data["lane_invasion_sensor"][1]
        }

        sensor_readings["sensor.camera.rgb/top"] = {'image': input_data["sensor.camera.rgb/top"][1]}

        #TODO Add camera data

        # for idx, k in enumerate(self.sensor_names):

        #     elif k=="lane_invasion_sensor":
        #         sensor_readings[k] = {'num_lane_intersections': self.sensors[k].num_laneintersections,\
        #                                 'out_of_road': self.sensors[k].out_of_road}
        #     elif 'camera' in k:
        #         if world_frame is None:
        #             print("No world frame found! Skipping reading from camera sensor!!")
        #         else:
        #             camera_image = self.sensors[k]._read_data(world_frame)
        #             sensor_readings[k] = {'image': camera_image}
        #     else:
        #         print("Uninitialized sensor!")

        return sensor_readings

    def get_npc_poses(self):

        actor_list = self.world.get_actors()
        actor_poses = np.zeros([len(actor_list), 4])
        for i, actor in enumerate(actor_list):
            # Don't include poses of ego vehicle, and of non-vehicle actors
            if actor.id == self.actor.id or "vehicle" not in actor.type_id:
                continue

            pose = actor.get_transform()
            speed = util.get_speed_from_velocity(actor.get_velocity())
            actor_poses[i, :] = np.array([pose.location.x, pose.location.y, pose.rotation.yaw, speed])

        return actor_poses

    def update_measurements(self, input_data):
        episode_measurements = {}

        ego_transform = self.actor.get_transform()
        ego_velocity = self.actor.get_velocity()
        speed = util.get_speed_from_velocity(ego_velocity)

        left_steer = self.actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FL_Wheel)
        right_steer = self.actor.get_wheel_steer_angle(carla.VehicleWheelLocation.FR_Wheel)
        steer_angle = (left_steer + right_steer) / (2 * 90)


        next_orientation, \
        dist_to_trajectory, \
        distance_to_goal_trajec, \
        self.next_waypoints, \
        next_wp_angles, \
        next_wp_vectors, \
        all_waypoints = self.global_planner.get_next_orientation_new(ego_transform)

        # ego vehicle features
        #TODO Add distance to goal
        episode_measurements.update({
            'ego_vehicle_location': [ego_transform.location.x, ego_transform.location.y, ego_transform.rotation.yaw],
            # 'ego_vehicle_velocity': ego_velocity,
            'speed': speed,
            'steer_angle': steer_angle,
            'dist_to_trajectory': dist_to_trajectory,
            'distance_to_goal_trajec' : distance_to_goal_trajec,
            'dist_to_goal' : ego_transform.location.distance(self.destination_transform.location),
            'next_orientation': next_orientation,
            # 'next_waypoints' : next_waypoints,
            'waypoints' : all_waypoints
        })

        episode_measurements["traffic_light"] = self.get_traffic_light_states()
        episode_measurements["obstacles"] = self.get_obstacle_states(self.next_waypoints)
        episode_measurements["dist_to_goal"] = self.actor.get_transform().location.distance(self.destination_transform.location)
        episode_measurements["autopilot_action"] = self.get_autopilot_action()

        # planner / waypoint features
        # next_orientation, dist_to_trajectory, dist_to_goal, next_waypoints, all_waypoints = self.planner.get_next_orientation_new(ego_transform)
        # episode_measurements.update({
        #     'next_orientation': next_orientation,
        #     'dist_to_trajectory': dist_to_trajectory,
        #     'dist_to_goal': dist_to_goal,
        #     'all_waypoints': all_waypoints
        # })
        # _, gps = input_data.get('GPS')
        # control features

        episode_measurements.update(self.get_sensor_readings(input_data))

        episode_measurements.update(self.current_controls)

        # Add NPC poses
        episode_measurements["npc_poses"] = self.get_npc_poses()

        # episode_measurements.update(self.get_sensor_readings(input_data))

        # # camera features
        # camera_features = {k: input_data.get(k) for k in input_data if 'camera' in k}
        # episode_measurements.update(camera_features)

        # # light features
        # light_features = self.get_traffic_light_features()
        # episode_measurements.update(light_features)

        # obstacle features
        # obstacle_features = self.get_obstacle_features(next_waypoints)
        # episode_measurements.update(obstacle_features)

        # symbolic features
        other_actors = self.world.get_actors().filter('*vehicle*')
        episode_measurements['symbolic_features'] = fetch_symbolic_dict(self.actor, other_actors, episode_measurements)

        return episode_measurements

    def run_step(self, input_data, timestamp):
        if self.step == 0:
            self.initialize(input_data)

        # # get episode measurements (features, rewards, images, etc...)
        ep_measurements = self.update_measurements(input_data)


        # # Write the data to the data buffer
        self.data_buffer["lock"].acquire()
        self.data_buffer['leaderboard_data'] = ep_measurements
        self.data_buffer["lock"].release()

        # # Set event to signal that data is ready
        # print(f"THREAD {self.step}: SENT DATA TO CARLA INTERFACE")
        self.receive_event.set()

        # # Wait for the carla_interface to send the action data
        self.send_event.wait()
        self.send_event.clear()
        # print(f"THREAD {self.step}: RECEIVED DATA FROM CARLA INTERFACE")

        # If a exit command is received, kill leaderboard evaluator
        if self.data_buffer["exit"]:
            raise KillSimulator("Received kill signal from main thread")
        # # If the policy sends a reset event, raise an exception to end the rollout
        if(self.data_buffer['reset']):
            self.step = 0
            self.data_buffer['reset'] = False
            print("THREAD: Resetting")
            raise Exception("Resetting")

        # # TODO WHY IS THIS HERE
        # # _, col = input_data.get('COLLISION')
        # # if col:
        # #     raise Exception('collision')

        self.step += 1

        return self.get_control(self.data_buffer["policy_action"])

    def get_control(self, action):
        """ Get Control object for Carla from action
        Input:
            - action: tuple containing (steer, throttle, brake) in [-1, 1]
        Output:
            - control: Control object for Carla
        """

        episode_measurements = {}

        if self.action_config.action_type != "control":
            action = action.flatten()

        if self.action_config.action_type is "sep_gas":
            steer = float(action[0])
            throttle = float(action[1])
            brake = float(action[2])

        elif self.action_config.action_type == "merged_speed_scaled_tanh":
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = (action[1] * 1.5) + 1
            target_speed = float(np.clip(target_speed * 10, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.action_config.action_type == "merged_speed_pid_test":
            # steer = float(action[0])
            steer = (float(action[0]))
            target_speed = float(action[1])
            current_speed = util.get_speed_from_velocity(self.actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0
        elif self.action_config.action_type == "merged_speed_tanh":
            # steer = float(action[0])
            steer = np.clip(float(action[0]), -1.0, 1.0)
            target_speed = float(np.clip((action[1] + 1) * 10.0, 0, self.target_speed))
            current_speed = util.get_speed_from_velocity(self.actor.get_velocity()) * 3.6
            gas = self.controller.pid_control(target_speed, current_speed, enable_brake=self.action_config.enable_brake)
            if gas < 0:
                throttle = 0.0
                brake = abs(gas)
            else:
                throttle = gas
                brake = 0.0

        elif self.action_config.action_type == "control":
            target_speed = -1
            episode_measurements["target_speed"] = target_speed
            return action, episode_measurements
        else:
            raise Exception("Invalid Action Type")

        return carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0)


    def get_traffic_light_states(self):
        ego_vehicle = self.basic_agent
        traffic_actors = self.world.get_actors().filter('*traffic_light*')
        traffic_actor, dist, traffic_light_orientation, nearest_light_transform = ego_vehicle.find_nearest_traffic_light(traffic_actors)


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

        ego_vehicle_actor = self.actor
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
                                        self.obs_config.vehicle_proximity_threshold)

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

    def get_hazard(self):
        return self.basic_agent.check_for_hazard()

    def get_autopilot_action(self, target_speed = 1.0):
        hazard_detected = self.get_hazard()

        if hazard_detected:
            return np.array([0,-1])
        else:
            waypoint = self.next_waypoints[2]
            steer = self.lateral_controller.pid_control(waypoint)
            steer = np.clip(steer, -1, 1)
            return np.array([steer, target_speed])
