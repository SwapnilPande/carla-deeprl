import numpy as np
from tqdm import tqdm
import scipy.spatial
from collections import deque
import itertools

import torch
import gym
from gym.spaces import Box, Discrete, Tuple

# compute reward
from projects.morel_mopo.algorithm.reward import compute_reward
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils
from projects.morel_mopo.algorithm.fake_envs import fake_env_scenarios as fescenarios
from projects.morel_mopo.algorithm.fake_envs.agent import ActorManager
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
        print(f'Input dim: {self.input_dim}, Output dim : {self.output_dim}')

        ################################################
        # Creating Action and Observation spaces
        ################################################
        self.action_space = self.config.action_config.action_space
        self.observation_space = self.config.obs_config.observation_space

        ################################################
        # Dataset comes from dynamics
        ################################################
        ## Testing a handcrafted scenario
        # self.offline_data_module = fescenarios.StraightWithStaticObstacle(self.frame_stack, self.norm_stats)#self.dynamics.data_module
        self.offline_data_module = self.dynamics.data_module
        if(self.offline_data_module is None):
            print("FAKE_ENV: Data module does not have associated dataset. Reset must be called with an initial state input.")

        ################################################
        # Actor Manager
        ################################################
        self.actor_manager = ActorManager(self.config,
                                          self.dynamics,
                                          self.device,
                                        )

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


    def reset(self, inp=None):
        ''' Resets environment. If no input is passed in, sample from dataset '''

        self.test = False
        # print("FAKE_ENV: Resetting environment...\n")

        if inp is None:
            # Raise error if no dataset is loaded AND no input is passed
            if(self.offline_data_module is None):
                raise Exception("FAKE_ENV: Cannot sample from dataset since dynamics model does not have associated dataset.")

            # Draw a sample from the dataset
            ((obs, action, _, _, _, ego_pose), waypoints, npc_poses) = self.sample()

            # Bring back traffic lights
            # try:
            #     self.traffic_light_locs = torch.stack(traffic_light_locs)
            #     self.traffic_light_states = torch.stack(traffic_light_states)
            # except:
            #     print("FAKE_ENV: No npc poses ------------------------------------------")
            #     self.traffic_light_locs = torch.empty(SAMPLE_STEPS, 0)
            #     self.traffic_light_states = torch.empty(SAMPLE_STEPS, 0)

        else:
            (obs, action, ego_pose, waypoints, npc_poses) = inp

        # Convert numpy arrays, or lists, to torch tensors
        obs = feutils.tensorify(obs, self.device)
        action = feutils.tensorify(action, self.device)
        ego_pose = feutils.tensorify(ego_pose, self.device)
        waypoints = feutils.tensorify(waypoints, self.device)
        npc_poses = feutils.tensorify(npc_poses, self.device)

        # This removes duplicate waypoints from the waypoints list'
        # TODO Revisit this - shouldn't be needed
        waypoints = feutils.filter_waypoints(waypoints).to(self.device)

        ## Remove any actors located at (0,0)
        npc_poses = feutils.remove_bad_actors(npc_poses)

        # Reset the actor_manager to instatiate the new agents
        self.actor_manager_state = self.actor_manager.reset(
            ego_pose,
            obs,
            action,
            waypoints,
            npc_poses
        )

        self.num_actors = self.actor_manager.num_actors

        # Save these in separate variables for easy access
        self.speeds = self.actor_manager_state['speeds']
        self.steers = self.actor_manager_state['steers']
        self.poses = self.actor_manager_state['poses']
        self.waypoints = self.actor_manager_state['waypoints']

        ## Instantiate the done array for all of the actors
        # We maintain a done array so that we can artificall delay the termination for a given actor
        # by only setting done true after a done_delay steps of being done for the actor
        # TODO: Move this to config
        self.dones_delay = 2
        self.dones = torch.zeros(self.num_actors, dtype=torch.bool)
        # Store the indices of the actors that are not done
        self.active_indices = torch.arange(self.num_actors).to(self.device)

        # done_counter is used to keep track of how many steps an actor has been done for
        self.dones_counter = torch.zeros(self.num_actors, dtype=torch.int)


        # TODO: Add handling of the last few waypoints back
        # if(len(self.waypoints) == 2):
        #     self.second_last_waypoint = self.waypoints[0]
        #     self.last_waypoint = self.waypoints[1]
        # else:
        #     self.second_last_waypoint = None

        # if(len(self.waypoints) == 1):
        #     self.last_waypoint = self.waypoints[0]
        # else:
        #     self.last_waypoint = None
        # self.previous_waypoint = None

        # TODO: Add traffic lights back
        # self.traffic_light_locs = self.traffic_light_locs.to(self.device)
        # self.traffic_light_states = self.traffic_light_states.to(self.device)

        self.steps_elapsed = 0
        self.episode_uncertainty = 0

        # Return policy observation for all actors
        return self.get_policy_obs().cpu().numpy()

    # def get_current_state(self):
    #     return (self.state.unnormalized,
    #             self.past_action.unnormalized,
    #             self.vehicle_pose,
    #             self.waypoints,
    #             self.npc_poses)

    def calc_disc(self, predictions):
        # Compute the pairwise distances between all predictions
        return scipy.spatial.distance_matrix(predictions, predictions)


    def usad(self, predictions):
        # thres = self.mean + (self.uncertain_threshold * self.beta_max) * (self.std)

        # max_discrep = np.amax(self.calc_disc(predictions)
        # If maximum is greater than threshold, return true
        return np.amax(self.calc_disc(predictions))

    def get_policy_obs_single_actor(self, agent_idx) -> torch.Tensor:

        # This agent is done, no need to compute policy obs
        # Return torch tensor of zeros
        if(self.dones[agent_idx]):
            return torch.zeros(1, self.observation_space.shape[-1]).to(self.device)


        angle, \
        dist_to_trajectory, \
        remaining_waypoints, \
        self.second_last_waypoint, \
        self.last_waypoint, \
        self.previous_waypoint = feutils.process_waypoints(self.waypoints[agent_idx],
                                                        self.poses[agent_idx],
                                                        self.device)
        #TODO: Fix handling of waypoints at the end of the route

                                 #                       second_last_waypoint = self.second_last_waypoint,
                                  #                      last_waypoint = self.last_waypoint,
                                                        #previous_waypoint = self.previous_waypoint)
              # next_waypoints,\
        # _, _, \


        # convert to tensors
        self.waypoints[agent_idx] = torch.FloatTensor(remaining_waypoints)
        dist_to_trajectory = torch.Tensor([dist_to_trajectory]).to(self.device)
        angle              = torch.Tensor([angle]).to(self.device)

        # Only for type hints to be happy
        out = torch.Tensor()

        if(self.config.obs_config.input_type == "wp_obs_info_speed_steer"):
            out = torch.cat([angle, self.speeds[agent_idx] / 10, self.steers[agent_idx], dist_to_trajectory], dim=0).float().to(self.device)#, dist_to_trajectory, angle



        elif(self.config.obs_config.input_type == "wp_obstacle_speed_steer"):
            #TODO Check config here
            obstacle_dist, obstacle_vel = self.get_obstacle_states(agent_idx)
            if(obstacle_dist == -1):
                obstacle_dist = 1
                obstacle_vel = 1
            else:
                obstacle_dist /= self.config.obs_config.vehicle_proximity_threshold
                obstacle_vel /= 20

            out = torch.cat([angle, self.speeds[agent_idx] / 10, self.steers[agent_idx], dist_to_trajectory, torch.tensor([obstacle_dist]).to(self.device), torch.tensor([obstacle_vel]).to(self.device)], dim=0).float().to(self.device)#dist_to_trajectory, angle

        elif(self.config.obs_config.input_type == "wp_obstacle_speed_steer"):
            obstacle_dist, obstacle_vel = self.get_obstacle_states(agent_idx)
            if(obstacle_dist == -1):
                obstacle_dist = 1
                obstacle_vel = 1
            else:
                obstacle_dist /= self.config.obs_config.vehicle_proximity_threshold
                obstacle_vel /= 20

            out = torch.cat([angle, self.speeds[agent_idx] / 10, self.steers[agent_idx], dist_to_trajectory, torch.tensor([obstacle_dist]).to(self.device), torch.tensor([obstacle_vel]).to(self.device)], dim=0).float().to(self.device) #dist_to_trajectory, angle

        # Make observation (1, obs_dim)
        return torch.unsqueeze(out, dim=0)

    def get_policy_obs(self) -> torch.tensor:
        # Compute the policy observation for all actors
        return torch.stack([self.get_policy_obs_single_actor(i) for i in range(self.num_actors)], dim=0)

    def check_collision(self, actor_1_pose, actor_2_pose):
        # This is cos(theta) from the centroid of our vehicle to one of the corners
        # any value larger than this signifies that the vehicle collision would be with the front of the car
        # any value smaller than this signifies that the vehicle collision would be with the back of the car
        # TODO we are not handling rear collision properly (Assumes that it is the same as side collision)
        car_corner_angle = 0.913
        front_collision_threshold = 5.5 #meters
        side_collision_threshold = 2 # meters

        # Vector for current heading of vehicle
        heading_vector_actor_1 = torch.tensor([torch.cos(torch.deg2rad(actor_1_pose[2])), torch.sin(torch.deg2rad(actor_1_pose[2]))]).to(self.device)
        heading_vector_actor_2 = torch.tensor([torch.cos(torch.deg2rad(actor_2_pose[2])), torch.sin(torch.deg2rad(actor_2_pose[2]))]).to(self.device)

        vec_to_other_vehicle_actor_1 = actor_2_pose[0:2] - actor_1_pose[0:2]
        vec_to_other_vehicle_actor_2 = -vec_to_other_vehicle_actor_1

        # Calculate normalized dot product - heading vector is a unit vector, so don't need to divide by that
        norm_dot_actor_1 = torch.dot(heading_vector_actor_1, vec_to_other_vehicle_actor_1) / torch.norm(vec_to_other_vehicle_actor_1)
        norm_dot_actor_2 = torch.dot(heading_vector_actor_2, vec_to_other_vehicle_actor_2) / torch.norm(vec_to_other_vehicle_actor_2)

        # Compute collision threshold based on heading of vehicle
        if(torch.abs(norm_dot_actor_1) > car_corner_angle and torch.abs(norm_dot_actor_2) > car_corner_angle):
            collision_threshold = front_collision_threshold
        elif(torch.abs(norm_dot_actor_1) > car_corner_angle or torch.abs(norm_dot_actor_2) > car_corner_angle):
            collision_threshold = side_collision_threshold
        else: # Both are colliding on the side
            collision_threshold = side_collision_threshold

        # Vector between the other vehicle and our vehicle
        vec_to_other_vehicle = actor_2_pose[0:2] - actor_1_pose[0:2]

        return torch.norm(vec_to_other_vehicle) < collision_threshold


    def get_obstacle_states(self, agent_idx):
        obstacle_state = {}
        obstacle_state['obstacle_visible'] = False
        obstacle_state['obstacle_orientation'] = -1

        min_obs_distance = 100000000
        found_obstacle = False

        cur_actor_pose = self.poses[agent_idx]
        next_waypoints = self.waypoints[agent_idx]

        # try:
        # Compute the distance to the closest obstacle for each active vehicle
        for i in self.active_indices:
            if(i == agent_idx):
                continue

            # if the object is not in our lane it's not an obstacle
            d_bool, distance = feutils.is_within_distance_ahead(self.poses[i], cur_actor_pose, self.config.obs_config.vehicle_proximity_threshold)

            if not d_bool:
                continue
            else:
                if not feutils.check_if_vehicle_in_same_lane(self.poses[i], next_waypoints, 1.3, self.device):
                    continue

                found_obstacle = True
                obstacle_state['obstacle_visible'] = True

                if distance < min_obs_distance:
                    obstacle_state['obstacle_dist'] = distance
                    obstacle_state['obstacle_speed'] = self.speeds[i]

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
    def step(self, new_actions):

        new_actions = feutils.tensorify(new_actions, self.device)

        # Clamp actions
        #TODO: Clamp according to action space
        new_actions = torch.clamp(
                        new_actions,
                        -1, 1
                    ).to(self.device)

        ## Step actor manager to get new states
        self.actor_manager_state = self.actor_manager.step(new_actions, self.dones)

        # Break actor_manager state into separate variables
        self.poses = self.actor_manager_state['poses']
        self.speeds = self.actor_manager_state['speeds']
        self.steers = self.actor_manager_state['steers']
        self.waypoints = self.actor_manager_state['waypoints']


        ###################### calculate waypoint features (in global frame)  ##############################
        policy_obs = self.get_policy_obs()

        ################## Calculate reward with penalty for uncertainty ##############################
        # Only need to compute collisions between every combination of vehicles
        # This halves computation
        # Get all combinations of actors
        collisions = torch.zeros(self.num_actors, dtype=torch.bool).to(self.device)

        # Only compute collisions for the active actors
        for actor_i, actor_j in itertools.combinations(self.active_indices, 2):
            # Check each collision independently
            side_collision = False
            front_collision = False

            collision = self.check_collision(self.poses[actor_i], self.poses[actor_j])

            # Store collision for each actor
            # If previously collided, continue to set collision to true
            collisions[actor_i] = collisions[actor_i] | collision
            collisions[actor_j] = collisions[actor_j] | collision

            # side_collision = side_collision or (collision and is_side)
            # front_collision = front_collision or (collision and not is_side)

        dist_to_trajectories = np.squeeze(policy_obs[..., 3]) * self.config.obs_config.vehicle_proximity_threshold

        out_of_lanes = torch.abs(dist_to_trajectories) > DIST

        #TODO: Change this back to when number of waypoints is 1
        # success = len(self.waypoints) == 1 or self.steps_elapsed >= len(self.npc_poses)
        successes = torch.tensor([len(waypoints) <= 3 for waypoints in self.waypoints])# or self.steps_elapsed >= len(self.npc_poses)

        active_rewards = compute_reward(np.squeeze(self.speeds[torch.logical_not(self.dones)]),
                                dist_to_trajectories[torch.logical_not(self.dones)],
                                torch.logical_or(collisions[torch.logical_not(self.dones)], out_of_lanes[torch.logical_not(self.dones)]),
                                self.config)

        # Construct full rewards array
        rewards = torch.zeros(self.num_actors).to(self.device)
        rewards[torch.logical_not(self.dones)] = active_rewards

        #TODO: Determine if uncertainty should be added
        # uncertain =  self.usad(self.deltas.normalized.detach().cpu().numpy())
        # reward_out[0] = reward_out[0]# - uncertain * self.uncertainty_coeff
        # reward_out = torch.squeeze(reward_out)
        # self.episode_uncertainty += uncertain

        # advance time
        self.steps_elapsed += 1
        self.cum_step += 1
        timeout = self.steps_elapsed >= self.timeout_steps

        # log
        # if(uncertain and self.logger is not None):
        #     # self.logger.get_metric("average_halt")
        #     self.logger.log_hyperparameters({"halts" : 1})
        # elif(timeout and self.logger is not None):
        #     self.logger.log_hyperparameters({"halts" : 0})


        ######################### build policy input ##########################

        # Policy input (unnormalized): dist_to_trajectory, next orientation, speed, steer
        # policy_input = torch.cat([dist_to_trajectory, angle, torch.flatten(self.state.unnormalized[0, :])], dim=0).float().to(self.device)

        # TODO: predictions should return ALL pose predictions, but currently only keeps
        # the sampled one
        info = {
                    # "delta" : self.deltas.normalized[self.model_idx].cpu().numpy(),
                    # "uncertain" : self.uncertainty_coeff * uncertain,
                    # "predictions": self.vehicle_pose.cpu().numpy(),
                    # "num_remaining_waypoints" : len(self.waypoints),
                }

        # Check done condition for this step
        cur_dones = torch.logical_or(successes, out_of_lanes.cpu())
        cur_dones = torch.logical_or(cur_dones, collisions.cpu())

        # For all actors that are not done, set done_counter to 0
        # Otherwise, increment done_counter
        self.dones_counter[torch.logical_not(cur_dones)] = 0
        self.dones_counter[cur_dones] += 1

        # If done_counter is greater than or equal to done_delay, set done to true
        # Done is latching, stays done once done
        self.dones = torch.logical_or(self.dones, self.dones_counter >= self.dones_delay)

        # Update active indices
        self.active_indices = torch.where(torch.logical_not(self.dones))[0]

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

        res = policy_obs.cpu().numpy(), rewards.cpu().numpy(), self.dones.cpu().numpy(), info

        #Logging
        # if(done and self.logger is not None):
        #     # Log how the episode terminated
        #     self.logger.log_scalar('rollout/obstacle_collision', int(collision), self.cum_step)
        #     self.logger.log_scalar('rollout/out_of_lane', int(out_of_lane), self.cum_step)
        #     self.logger.log_scalar('rollout/success', int(success), self.cum_step)
        #     # Log average uncertainty over the episode
        #     self.logger.log_scalar('rollout/average_uncertainty', self.episode_uncertainty / self.steps_elapsed, self.cum_step)

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
