from re import S
import numpy as np
from numpy.lib.arraysetops import isin
from tqdm import tqdm
import scipy.spatial
from collections import deque

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import gym
from gym.spaces import Box, Discrete, Tuple

# compute reward
from projects.morel_mopo.algorithm.reward import compute_reward
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils
from projects.morel_mopo.algorithm.fake_envs import fake_env_scenarios as fescenarios
DIST = 0.5


class PIDLateralController():
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, K_P=1.0, K_D=0.0, K_I=0.0, dt=0.03):
        """
        :param K_P: Proportional term
        :param K_D: Differential term
        :param K_I: Integral term
        :param dt: time differential in seconds
        """
        self._K_P = K_P
        self._K_D = K_D
        self._K_I = K_I
        self._dt = dt
        self._e_buffer = deque(maxlen=10)

    def pid_control(self, vehicle_pose, waypoint):
        """
        Estimate the steering angle of the vehicle based on the PID equations

        :param waypoint: target waypoint
        :param vehicle_transform: current transform of the vehicle
        :return: steering control in the range [-1, 1]
        """
        v_begin = vehicle_pose[0:2]
        theta = vehicle_pose[2]

        v_end = v_begin + np.array([np.cos(np.radians(theta)),
                                         np.sin(np.radians(theta))])

        v_vec = np.array([v_end[0] - v_begin[0], v_end[1] - v_begin[1], 0.0])
        w_vec = np.array([waypoint[0] -
                          v_begin[0],
                          waypoint[1] -
                          v_begin[1], 0.0])
        _dot = np.arccos(np.clip(np.dot(w_vec, v_vec) /
                                 (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

        _cross = np.cross(v_vec, w_vec)
        if _cross[2] < 0:
            _dot *= -1.0

        self._e_buffer.append(_dot)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._K_P * _dot) + (self._K_D * _de /
                                             self._dt) + (self._K_I * _ie * self._dt), -1.0, 1.0)


class BaseFakeEnv(gym.Env):
    def __init__(self, dynamics,
                        config,
                        policy_data_module_config = None,
                        logger = None):

        # Save logger
        self.logger = logger

        # Save and verify config
        self.config = config
        self.config.verify()



        ################################################
        # Dynamics parameters
        ################################################
        self.dynamics = dynamics

        # device
        self.device_num = self.dynamics.gpu
        self.device = "cuda:{}".format(self.device_num) if torch.cuda.is_available() else "cpu"
        print('Device: ', self.device)
        self.dynamics.to(self.device)

        self.input_dim = self.dynamics.state_dim_in
        self.output_dim = self.dynamics.state_dim_out
        self.action_dim = self.dynamics.action_dim
        print(f'Input dim: {self.input_dim}, Output dim: {self.output_dim}')

        ################################################
        # Creating Action and Observation spaces
        ################################################
        self.action_space = self.config.action_config.action_space
        self.observation_space = self.config.obs_config.observation_space

        self.state = None
        self.vehicle_pose = None
        self.waypoints = None

        ################################################
        # Dataset comes from dynamics
        ################################################
        # Get norm stats and frame_stack from dynamics
        # This will be the same as the norm stats from the original dataset, on which the dynamics model was trained
        self.norm_stats = self.dynamics.normalization_stats
        self.frame_stack = self.dynamics.frame_stack

        # self.offline_data_module = fescenarios.StraightWithStaticObstacle(self.frame_stack, self.norm_stats)#self.dynamics.data_module
        self.policy_data_module_config = policy_data_module_config
        if(self.policy_data_module_config is not None):
            print("FAKE ENV: Loading policy training dataset")
            self.offline_data_module = self.policy_data_module_config.dataset_type(self.policy_data_module_config, self.norm_stats)
            self.offline_data_module.setup()
        else:
            print("No policy training dataset passed, using dataset from dynamics")
            self.offline_data_module = self.dynamics.data_module
        if(self.offline_data_module is None):
            print("FAKE_ENV: Data module does not have associated dataset. Reset must be called with an initial state input.")

        ################################################
        # State variables for the fake env
        ################################################
        self.state = feutils.NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], self.device)
        self.past_action = feutils.NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], self.device)
        self.deltas = feutils.NormalizedTensor(self.norm_stats["delta"]["mean"], self.norm_stats["delta"]["std"], self.device)

        ################################################
        # MOPO hyperparameters
        ################################################
        self.uncertainty_coeff = self.config.uncertainty_coeff
        self.timeout_steps = self.config.timeout_steps

        if(self.logger is not None):
            self.logger.log_scalar("mopo/uncertainty_coeff", self.uncertainty_coeff)
            self.logger.log_scalar("mopo/rollout_length", self.timeout_steps)


        # ONLY FOR EXPERT AUTOPILOT
        self.args_lateral_dict = {
            'K_P': 0.88,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1/10.0}
        self.lateral_controller = PIDLateralController(K_P=self.args_lateral_dict['K_P'], K_D=self.args_lateral_dict['K_D'], K_I=self.args_lateral_dict['K_I'], dt=self.args_lateral_dict['dt'])

        # Cumulative steps across all training runs
        self.cum_step = 0
        self.prev_termination_log = 0


    # sample from dataset
    def sample(self):
        return self.offline_data_module.sample_with_waypoints(self.timeout_steps)

    ''' Resets environment. If no input is passed in, sample from dataset '''
    def reset(self, inp=None):

        self.test = False
        # print("FAKE_ENV: Resetting environment...\n")

        if inp is None:
            if(self.offline_data_module is None):
                raise Exception("FAKE_ENV: Cannot sample from dataset since dynamics model does not have associated dataset.")
            ((obs, action, _, _, _, vehicle_pose), waypoints, npc_poses) = self.sample()
            self.state.normalized = obs[:,:2]
            self.past_action.normalized = action
            try:
                if(not isinstance(npc_poses, torch.Tensor)):
                    self.npc_poses = torch.stack(npc_poses)

                else:
                    self.npc_poses = npc_poses
            except:
                self.npc_poses = torch.empty(self.timeout_steps, 0)

            # Bring back traffic lights
            # try:
            #     self.traffic_light_locs = torch.stack(traffic_light_locs)
            #     self.traffic_light_states = torch.stack(traffic_light_states)
            # except:
            #     print("FAKE_ENV: No npc poses ------------------------------------------")
            #     self.traffic_light_locs = torch.empty(SAMPLE_STEPS, 0)
            #     self.traffic_light_states = torch.empty(SAMPLE_STEPS, 0)

        else:
            (obs, action, vehicle_pose, waypoints, npc_poses) = inp

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(obs, torch.Tensor)):
                obs = torch.FloatTensor(obs)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(action, torch.Tensor)):
                action = torch.FloatTensor(action)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(vehicle_pose, torch.Tensor)):
                vehicle_pose = torch.FloatTensor(vehicle_pose)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(waypoints, torch.Tensor)):
                waypoints = torch.FloatTensor(waypoints)

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(waypoints, torch.Tensor)):
                waypoints = torch.FloatTensor(waypoints)

            # state only includes speed, steer
            self.state.unnormalized =  obs[:,:2]
            self.past_action.unnormalized = action
            self.npc_poses = npc_poses

        self.waypoints = feutils.filter_waypoints(waypoints).to(self.device)

        if(not isinstance(self.npc_poses, torch.Tensor)):
            self.npc_poses = torch.Tensor(self.npc_poses).to(self.device)
        remove_indices = []
        # Loop over all poses at the first time step
        for i in range(self.npc_poses.shape[1]):
            if(torch.all(self.npc_poses[0,i] == torch.zeros(self.npc_poses[0,i].shape)).to(self.device)):
                remove_indices.append(i)
                # print(f"Removing {i}\t {self.npc_poses[0,i]}")
        if(len(remove_indices) > 0):
            remove_mask = np.ones(self.npc_poses.shape[1], dtype=bool)
            remove_mask[remove_indices] = False
            self.npc_poses = self.npc_poses[:, remove_mask, :]
            self.npc_poses = self.npc_poses.to(self.device)
        else:
            self.npc_poses = self.npc_poses.to(self.device)


        if(len(self.waypoints) == 2):
            self.second_last_waypoint = self.waypoints[0]
            self.last_waypoint = self.waypoints[1]
        else:
            self.second_last_waypoint = None

        if(len(self.waypoints) == 1):
            self.last_waypoint = self.waypoints[0]
        else:
            self.last_waypoint = None
        self.previous_waypoint = None

        # self.traffic_light_locs = self.traffic_light_locs.to(self.device)
        # self.traffic_light_states = self.traffic_light_states.to(self.device)

        self.vehicle_pose = vehicle_pose.to(self.device)

        self.steps_elapsed = 0
        self.episode_uncertainty = 0

        self.model_idx = np.random.choice(self.dynamics.n_models)
        # Reset hidden state

        #TODO Return policy features, not dynamics features
        policy_obs, _, _ = self.get_policy_obs(self.state.unnormalized[0])


        return policy_obs.cpu().numpy()

    def get_current_state(self):
        return (self.state.unnormalized,
                self.past_action.unnormalized,
                self.vehicle_pose,
                self.waypoints,
                self.npc_poses)


    def calc_disc(self, predictions):
        # Compute the pairwise distances between all predictions
        return scipy.spatial.distance_matrix(predictions, predictions)


    def usad(self, predictions):
        # thres = self.mean + (self.uncertain_threshold * self.beta_max) * (self.std)

        # max_discrep = np.amax(self.calc_disc(predictions)
        # If maximum is greater than threshold, return true
        return np.amax(self.calc_disc(predictions))


    '''
    Updates state vector according to dynamics prediction
    # @params: delta      [Δspeed, Δsteer]
    # @return new state:  [[speed_t+1, steer_t+1], [speed_t, steer_t], speed_t-1, steer_t-1]]
    '''
    def update_next_state(self, past_state, delta_state):
        raise NotImplementedError

    def update_action(self, prev_action, new_action):
        raise NotImplementedError

    def make_prediction(self, past_state, past_action):
        raise NotImplementedError

    def get_policy_obs(self, state):
        angle, \
        dist_to_trajectory, \
        next_waypoints,\
        _, _, \
        remaining_waypoints, \
        self.second_last_waypoint, \
        self.last_waypoint, \
        self.previous_waypoint = feutils.process_waypoints(self.waypoints,
                                                        self.vehicle_pose,
                                                        self.device,
                                                        second_last_waypoint = self.second_last_waypoint,
                                                        last_waypoint = self.last_waypoint,
                                                        previous_waypoint = self.previous_waypoint)

        # convert to tensors
        self.waypoints = torch.FloatTensor(remaining_waypoints)
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
        angle              = torch.Tensor([angle]).to(self.device)


        if(self.config.obs_config.input_type == "wp_obs_info_speed_steer"):
            return torch.cat([angle, state[1:2] / 10, state[0:1], dist_to_trajectory], dim=0).float().to(self.device), dist_to_trajectory, angle



        elif(self.config.obs_config.input_type == "wp_obstacle_speed_steer"):
            #TODO Check config here
            cur_npc_poses = self.npc_poses[self.steps_elapsed]
            obstacle_dist, obstacle_vel = self.get_obstacle_states(cur_npc_poses, self.waypoints)
            if(obstacle_dist == -1):
                obstacle_dist = 1
                obstacle_vel = 1
            else:
                obstacle_dist /= self.config.obs_config.vehicle_proximity_threshold
                obstacle_vel /= 20

            return torch.cat([angle, state[1:2] / 10, state[0:1], dist_to_trajectory, torch.tensor([obstacle_dist]).to(self.device), torch.tensor([obstacle_vel]).to(self.device)], dim=0).float().to(self.device), dist_to_trajectory, angle

        elif(self.config.obs_config.input_type == "wp_360_obstacle_speed_steer"):
            cur_npc_poses = self.npc_poses[self.steps_elapsed]

            # Initialize obstacle outputs
            front_obs_vec = torch.tensor([1.5, 1.5]).to(self.device)
            front_obs_vel = torch.tensor([1.5, 1.5]).to(self.device)
            front_min_dist = 10000

            front_right_obs_vec = torch.tensor([1.5, 1.5]).to(self.device)
            front_right_obs_vel = torch.tensor([1.5, 1.5]).to(self.device)
            front_right_min_dist = 10000

            front_left_obs_vec = torch.tensor([1.5, 1.5]).to(self.device)
            front_left_obs_vel = torch.tensor([1.5, 1.5]).to(self.device)
            front_left_min_dist = 10000

            back_right_obs_vec = torch.tensor([1.5, 1.5]).to(self.device)
            back_right_obs_vel = torch.tensor([1.5, 1.5]).to(self.device)
            back_right_min_dist = 10000

            back_left_obs_vec = torch.tensor([1.5, 1.5]).to(self.device)
            back_left_obs_vel = torch.tensor([1.5, 1.5]).to(self.device)
            back_left_min_dist = 10000



            # Get distance between ego vehicle and all other vehicles
            vehicle_vectors = cur_npc_poses[...,0:2] - self.vehicle_pose[0:2]
            distances = torch.norm(vehicle_vectors, dim=1)
            # Next, get the indices for all distances less than the threshold
            in_range_npcs = (distances < self.config.obs_config.vehicle_proximity_threshold).nonzero()

            # Vehicles in range
            if(in_range_npcs.shape[0] > 0):

                in_range_npcs = in_range_npcs[0]
                # Get poses for all vehicles in range
                in_range_npc_poses = cur_npc_poses[in_range_npcs]
                in_range_distances = distances[in_range_npcs]

                # Construct ego H_transform
                ego_frame_rot = feutils.rot(self.vehicle_pose[2])
                ego_H_transform = torch.eye(3).to(self.device)
                ego_H_transform[0:2, 2] = self.vehicle_pose[0:2]
                ego_H_transform[0:2, 0:2] = ego_frame_rot

                # Construct H_transform for all vehicles
                # Stack of len(in_range_npcs) 3x3 matrices
                in_range_H_transform = torch.eye(3).to(self.device).repeat(in_range_npcs.shape[0], 1, 1)
                in_range_H_transform[:, 0:2, 2] = in_range_npc_poses[:, 0:2]
                in_range_H_transform[:, 0:2, 0:2] = feutils.rot(in_range_npc_poses[:, 2])

                # Compute vector of all vehicles in ego frame
                in_range_vehicle_vectors_in_ego_frame = torch.matmul(torch.inverse(ego_H_transform), in_range_H_transform)

                relative_rotations = in_range_vehicle_vectors_in_ego_frame[..., 0:2, 0:2]
                relative_positions = in_range_vehicle_vectors_in_ego_frame[..., 0:2, 2]
                relative_positions_normalized = relative_positions / torch.norm(relative_positions, dim=1, keepdim=True)
                relative_velocities = relative_rotations[:,0:2, 0] * in_range_npc_poses[:, 3]

                # Dot product is simply the first element of the normalized vector
                dot_products = relative_positions_normalized[...,0]

                # Finally, compute observations
                for index, (dot_product, distance) in enumerate(zip(dot_products, in_range_distances)):
                    # Obstacle is in front of vehicle
                    if dot_product > 0.995 and distance < front_min_dist:
                        front_min_dist = distance
                        front_obs_vec = relative_positions[index] / self.config.obs_config.vehicle_proximity_threshold
                        front_obs_vel = relative_velocities[index] / 20

                    # Obstacle is in front right
                    elif dot_product > 0 and relative_positions[index][1] > 0 and distance < front_right_min_dist:
                        front_right_min_dist = distance
                        front_right_obs_vec = relative_positions[index] / self.config.obs_config.vehicle_proximity_threshold
                        front_right_obs_vel = relative_velocities[index] / 20

                    # Obstacle is in front left
                    elif dot_product > 0 and relative_positions[index][1] < 0 and distance < front_left_min_dist:
                        front_left_min_dist = distance
                        front_left_obs_vec = relative_positions[index] / self.config.obs_config.vehicle_proximity_threshold
                        front_left_obs_vel = relative_velocities[index] / 20

                    # Obstacle is in back right
                    elif dot_product <= 0 and relative_positions[index][1] > 0 and distance < back_right_min_dist:
                        back_right_min_dist = distance
                        back_right_obs_vec = relative_positions[index] / self.config.obs_config.vehicle_proximity_threshold
                        back_right_obs_vel = relative_velocities[index] / 20

                    # Obstacle is in back left
                    elif dot_product <= 0 and relative_positions[index][1] < 0 and distance < back_left_min_dist:
                        back_left_min_dist = distance
                        back_left_obs_vec = relative_positions[index] / self.config.obs_config.vehicle_proximity_threshold
                        back_left_obs_vel = relative_velocities[index] / 20

            return torch.cat(
                [
                    angle,
                    state[1:2] / 10,
                    state[0:1],
                    dist_to_trajectory,
                    front_obs_vec[0:1],
                    front_obs_vec[1:2],
                    front_obs_vel[0:1],
                    front_obs_vel[1:2],
                    front_right_obs_vec[0:1],
                    front_right_obs_vec[1:2],
                    front_right_obs_vel[0:1],
                    front_right_obs_vel[1:2],
                    front_left_obs_vec[0:1],
                    front_left_obs_vec[1:2],
                    front_left_obs_vel[0:1],
                    front_left_obs_vel[1:2],
                    back_right_obs_vec[0:1],
                    back_right_obs_vec[1:2],
                    back_right_obs_vel[0:1],
                    back_right_obs_vel[1:2],
                    back_left_obs_vec[0:1],
                    back_left_obs_vec[1:2],
                    back_left_obs_vel[0:1],
                    back_left_obs_vel[1:2],
                ]
            ).float(), dist_to_trajectory, angle








    def check_collision(self, other_vehicle_pose):
        # This is cos(theta) from the centroid of our vehicle to one of the corners
        # any value larger than this signifies that the vehicle collision would be with the front of the car
        # any value smaller than this signifies that the vehicle collision would be with the back of the car
        # TODO we are not handling rear collision properly (Assumes that it is the same as side collision)
        car_corner_angle = 0.913
        front_collision_threshold = 5.5 #meters
        side_collision_threshold = 2 # meters

        # Vector for current heading of vehicle
        heading_vector = torch.tensor([torch.cos(torch.deg2rad(self.vehicle_pose[2])), torch.sin(torch.deg2rad(self.vehicle_pose[2]))]).to(self.device)

        # Vector between the other vehicle and our vehicle
        vec_to_other_vehicle = other_vehicle_pose[0:2] - self.vehicle_pose[0:2]

        # Calculate normalized dot product - heading vector is a unit vector, so don't need to divide by that
        norm_dot = torch.dot(heading_vector, vec_to_other_vehicle) / torch.norm(vec_to_other_vehicle)


        if(norm_dot > car_corner_angle):
            front_collision = torch.norm(vec_to_other_vehicle) < front_collision_threshold
            return torch.norm(vec_to_other_vehicle) < front_collision_threshold, False
        else:
            side_collision = torch.norm(vec_to_other_vehicle) < side_collision_threshold
            return torch.norm(vec_to_other_vehicle) < side_collision_threshold, True


    def get_obstacle_states(self, cur_npc_poses, next_waypoints):
        obstacle_state = {}
        obstacle_state['obstacle_visible'] = False
        obstacle_state['obstacle_orientation'] = -1

        min_obs_distance = 100000000
        found_obstacle = False

        # try:
        for i in range(cur_npc_poses.shape[0]):

            # if the object is not in our lane it's not an obstacle
            d_bool, distance = feutils.is_within_distance_ahead(cur_npc_poses[i], self.vehicle_pose, self.config.obs_config.vehicle_proximity_threshold)

            if not d_bool:
                continue
            else:
                if not feutils.check_if_vehicle_in_same_lane(cur_npc_poses[i], next_waypoints, 1.3, self.device):
                    continue

                found_obstacle = True
                obstacle_state['obstacle_visible'] = True

                if distance < min_obs_distance:
                    obstacle_state['obstacle_dist'] = distance
                    obstacle_state['obstacle_speed'] = cur_npc_poses[i][3]

                    min_obs_distance = distance

        if not found_obstacle:
            obstacle_state['obstacle_dist'] = -1
            obstacle_state['obstacle_speed'] = -1

        return obstacle_state['obstacle_dist'], obstacle_state['obstacle_speed']



    '''
    Takes one step according to action
    @ params: new_action
    @ returns next_obs, reward_out, (uncertain or timeout), {"delta" : delta, "uncertain" : 100*uncertain}
    '''
    def step(self, new_action):
        # print('Stepping with action', new_action)
        with torch.no_grad():

            # Convert numpy arrays, or lists, to torch tensors
            if(not isinstance(new_action, torch.Tensor)):
                new_action = torch.FloatTensor(new_action)

            # clamp new action to safe range
            #TODO Switch to using action space to clamp
            new_action = torch.squeeze(torch.clamp(new_action, -1, 1)).to(self.device)

            self.past_action.unnormalized = self.update_action(self.past_action, new_action)

            ############ feed obs, action into dynamics model for prediction ##############

            # input [[speed_t, steer_t, Δtime_t, action_t], [speed_t-1, steer_t-1, Δt-1, action_t-1]]
            # unsqueeze to form batch dimension for dynamics input

            # print(f'Curr vehicle pose: {self.vehicle_pose}')

            # print(f'State: {self.state.unnormalized}, Action: {self.past_action.unnormalized}')

            # Delta: prediction from one randomly selected model
            # [Δx_t+1, Δy_t+1, Δtheta_t+1, Δspeed_t+1, Δsteer_t+1]
            self.deltas.normalized = torch.clone(self.make_prediction(self.state, self.past_action))

            # predicted change in x, y, th
            delta_vehicle_poses = self.deltas.unnormalized[:,:3]


            # change in steer, speed
            delta_state        = self.deltas.unnormalized[self.model_idx,3:5]

            # update vehicle pose
            vehicle_loc_delta = torch.transpose(torch.tensordot(feutils.rot(torch.deg2rad(self.vehicle_pose[2])).to(self.device), delta_vehicle_poses[:,0:2], dims = ([1], [1])), 0, 1)

            # print("FAKE DELTA: {:4f} {:4f} {:4f}".format(vehicle_loc_delta.squeeze()[0].cpu().item(),
            #                                             vehicle_loc_delta.squeeze()[1].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[2].cpu().item()))
            # print("FAKE TRANSFORMED: {:4f} {:4f} {:4f}".format(delta_vehicle_poses.squeeze()[0].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[1].cpu().item(),
            #                                             delta_vehicle_poses.squeeze()[2].cpu().item()))

            vehicle_loc = self.vehicle_pose[0:2] + vehicle_loc_delta
            vehicle_rot = self.vehicle_pose[2] + delta_vehicle_poses[:,2]
            vehicle_rot = torch.unsqueeze(vehicle_rot, dim = 1)

            vehicle_poses = torch.cat([vehicle_loc, vehicle_rot], dim = 1)
            self.vehicle_pose = vehicle_poses[self.model_idx]

            if(self.vehicle_pose[2] < -180):
                self.vehicle_pose[2] += 360
            elif(self.vehicle_pose[2] > 180):
                self.vehicle_pose[2] -= 360

            # update next state
            self.state.unnormalized = self.update_next_state(self.state, delta_state)

            ###################### calculate waypoint features (in global frame)  ##############################
            policy_obs, dist_to_trajectory, angle = self.get_policy_obs(self.state.unnormalized[0])

            ################## calc reward with penalty for uncertainty ##############################
            collision = False
            cur_npc_poses = self.npc_poses[self.steps_elapsed]


            # Check each collision independently
            side_collision = False
            front_collision = False

            for other_vehicle_pose in cur_npc_poses:
                collision, is_side = self.check_collision(other_vehicle_pose)

                side_collision = side_collision or (collision and is_side)
                front_collision = front_collision or (collision and not is_side)

            out_of_lane = torch.abs(dist_to_trajectory) > DIST

            success = len(self.waypoints) <= 1 or (self.steps_elapsed >= len(self.npc_poses) - 1)

            # Compute velocity along trajectory
            trajectory_velocity = feutils.compute_trajectory_velocity(self.waypoints,
                                                                        self.vehicle_pose,
                                                                        self.state.unnormalized[0,1],
                                                                        self.device,
                                                                        second_last_waypoint = self.second_last_waypoint,
                                                                        last_waypoint = self.last_waypoint,
                                                                        previous_waypoint = self.previous_waypoint)
            reward_out = compute_reward(trajectory_velocity, dist_to_trajectory, side_collision or front_collision, out_of_lane, self.config)

            uncertain =  self.usad(self.deltas.normalized.detach().cpu().numpy())
            reward_out[0] = reward_out[0] - uncertain * self.uncertainty_coeff
            reward_out = torch.squeeze(reward_out)
            self.episode_uncertainty += uncertain

            # advance time
            self.steps_elapsed += 1
            self.cum_step += 1
            timeout = self.steps_elapsed >= self.timeout_steps

            # log
            if(uncertain and self.logger is not None):
                # self.logger.get_metric("average_halt")
                self.logger.log_hyperparameters({"halts" : 1})
            elif(timeout and self.logger is not None):
                self.logger.log_hyperparameters({"halts" : 0})


            ######################### build policy input ##########################

            # Policy input (unnormalized): dist_to_trajectory, next orientation, speed, steer
            # policy_input = torch.cat([dist_to_trajectory, angle, torch.flatten(self.state.unnormalized[0, :])], dim=0).float().to(self.device)

            info = {
                        "delta" : self.deltas.normalized[self.model_idx].cpu().numpy(),
                        "uncertain" : self.uncertainty_coeff * uncertain,
                        "predictions": vehicle_poses.cpu().numpy(),
                        "num_remaining_waypoints" : len(self.waypoints),
                    }

            # Check done condition
            # done = (success or
                    # out_of_lane or
                    # collision or
                    # self.steps_elapsed >= len(self.npc_poses))
            done = (success or
                    out_of_lane or
                    front_collision or
                    side_collision)


            # If done, print the termination type
            # if(done):
            #     # if(timeout):t
            #     #     print("Timeout")
            #     #     info["termination"] = "timeout"
            #     if(out_of_lane):
            #         print("Out of lane")
            #         info["termination"] = "out_of_lane"
            #     elif(front_collision):
            #         print("Collision")
            #         info["termination"] = "front_collision"
            #     elif(side_collision):
            #         print("Side Collision")
            #         info["termination"] = "side_collision"
            #     elif(success):
            #         print("Success")
            #         info["termination"] = "Success"
            #     elif(self.steps_elapsed >= len(self.npc_poses)):
            #         print("Not enough NPC poses")
            #         info["termination"] = "NPC"

            #     else:
            #         print("Unknown")
            #         info["termination"] = "unknown"
            #         print(len(self.npc_poses))

            try:
                res = policy_obs.cpu().numpy(), float(reward_out.item()), done, info
            except:
                import ipdb; ipdb.set_trace()

            #Logging
            if(done and self.logger is not None):
                # Log how the episode terminated
                self.logger.log_scalar('rollout/obstacle_collision', int(front_collision or side_collision), self.cum_step)
                self.logger.log_scalar('rollout/out_of_lane', int(out_of_lane), self.cum_step)
                self.logger.log_scalar('rollout/success', int(success), self.cum_step)
                # Log average uncertainty over the episode
                self.logger.log_scalar('rollout/average_uncertainty', self.episode_uncertainty / self.steps_elapsed, self.cum_step)

            return res

    def get_autopilot_action(self, target_speed=0.5):
        waypoint = self.waypoints[0]

        steer = self.lateral_controller.pid_control(self.vehicle_pose.cpu().numpy(), waypoint)
        steer = np.clip(steer, -1, 1)


        obstacle_dist = 1.0
        obstacle_vel = 1.0
        cur_npc_poses = self.npc_poses[self.steps_elapsed]
        for i in range(cur_npc_poses.shape[0]):
            d_bool, norm_target = feutils.is_within_distance_ahead(cur_npc_poses[i], self.vehicle_pose, self.config.obs_config.vehicle_proximity_threshold)

            if(d_bool):
                obstacle_dist = norm_target/self.config.obs_config.vehicle_proximity_threshold
                obstacle_vel = cur_npc_poses[i][3] / 20

        if(obstacle_dist < 0.3):
            target_speed = -1.0

        return np.array([steer, target_speed])