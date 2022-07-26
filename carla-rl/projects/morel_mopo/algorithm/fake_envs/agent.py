from collections import OrderedDict
import projects.morel_mopo.algorithm.fake_envs.fake_env_utils as feutils

import numpy as np
import cv2
import torch
import time
import matplotlib.pyplot as plt
# from common.debug_code import DebugCode


# TOWN-SPECIFIC MAP SETTINGS
SCALE = 1.0
PIXELS_PER_METER = 12.0
WORLD_OFFSET = {
    'Town01': (-52.059906005859375, -52.04996085166931),
    'Town02': (-57.45972919464111, 55.3907470703125),
    'Town03': (-199.0638427734375, -259.27125549316406),
    'Town04': (-565.26904296875, -446.1461181640625),
    'Town05': (-326.0445251464844, -257.8750915527344)
}



def delta_update_func(deltas: feutils.NormalizedTensor):
    """Preprocess delta before they are applied to the state

    Args:
        deltas (feutils.NormalizedTensor): Torch tensor containing the delta to apply to the state [n_agents, state_dim_out]

    Returns:
        state_delta (feutils.NormalizedTensor): Torch tensor containing the delta to apply to the state [n_agents, state_dim_in]
        additional_state_output: Additional output from the delta update function
            In this case, it contains the unnormalized delta pose of the vehicle
    """

    return deltas[..., 3:5], deltas.unnormalized[..., :3]

class ActorManager():
    """Class to handle the heteregeneous set of actors that we need to maintain the base_fake env.

    This class is designed to batch computation as much as possible for rolling out the various actors in the environment. The actor manager will also maintain the buffer of available policies to sample from.
    """

    def __init__(self, config, dynamics, device):

        self.config = config
        self.dynamics = dynamics
        self.frame_stack = self.dynamics.frame_stack
        self.device = device

        # Get the prediction wrapper for the dynamics model
        self.prediction_wrapper = self.dynamics.get_prediction_wrapper(delta_update_func)

        # Dictionary to store the various actors to roll out
        # Each entry in the list is a HomogeneousActorGroup object that handles all actors of the same type
        self.actor_groups = []

        # Sampling probabilities for each type of actor
        self.actor_type_probabilities = []

        # num_actors stores the total number of actors
        self.num_actors = 0

        # For rendering
        self.fig = None

        self.i = 0


    def reset(self,
                ego_pose,
                ego_states,
                ego_actions,
                ego_waypoints,
                actor_trajectories):

        """Reset the actor manager to the initial state of the new episode.

        Note that after initial reset, the actor manager treats the ego and npc vehicles
        equivalently. The input distinction only exists because we receive the state
        of the ego differently than we do the state of the NPCS

        Args:
            ego_pose: Initial pose of the ego vehicle
            ego_states: Initial state of the ego vehicle
            ego_actions: Initial actions of the ego vehicle
            ego_waypoints: Initial waypoints of the ego vehicle
            actor_trajectories: Torch tensor containing the states of the pose and velocity
                                 for the entire rollout
                                [time_steps, num_actors, 4 (x,y,theta,speed)]
        """
        self.i = 0

        # Additional + 1 for the ego vehicle
        self.num_actors = actor_trajectories.shape[1] + 1

        ## Reset the dynamics ensemble with initial states and actions
        # State input : [frame_stack, 2]
        #       Speed
        #       Steer
        # We don't actually have the steer for the autopilot actors
        # Dumb solution to this is to initialize actors to 0 speed and steer
        # TODO : Figure out a better solution
        # Reset the state of all of the actors
        initial_states = torch.zeros(self.num_actors,
                                    self.frame_stack,
                                    self.dynamics.state_dim_in,
                                    device=self.device)
        initial_states[..., -1] = torch.rand(*initial_states[..., -1].shape) / 5
        # Save the ego state in the last position
        # Unnormalize state before saving it, using the NormalizedTensor in prediction_wrapper
        initial_states[-1] = self.prediction_wrapper.states.unnormalize_array(ego_states)

        # Save the initial pose of all the actors
        # The initial pose of the actors is simply the first time step in the trajectories
        # Don't include the 4 element for each actor, as this is the actor velocity, which we are ignoring
        # and setting to 0
        self.poses = torch.zeros(
            self.num_actors, 3
        ).to(self.device)

        # Save the initial pose of the actors as the current pose
        self.poses[:-1] = actor_trajectories[0, :, :3]

        # Save the pose as the ego vehicle
        self.poses[-1] = ego_pose

        # If the current speed/steer is 0, then the actions should also be 0, this is actually -1 for the target_speed
        initial_actions = torch.zeros(self.num_actors,
                                          self.frame_stack,
                                          self.dynamics.action_dim,
                                          device=self.device)

        # Set the target_speed to -1 for all actors - this corresponds to full braking
        # initial_actions[...,1] = -1

        # Save initial actions for ego vehicle
        # Unnormalize state before saving it, using the NormalizedTensor in prediction_wrapper
        initial_actions[-1] = self.prediction_wrapper.actions.unnormalize_array(ego_actions)

        ## Generate waypoints for all of the actors
        # We generate waypoints by sparsifying the actor trajectories
        # For the ego vehicle, we are given the waypoints directly

        # Generate waypoints for the NPC vehicles
        # Subtract one because this doesn't generate teh waypoints for the ego vehicle
        self.waypoints = [feutils.get_waypoints(actor_trajectories[:,i,:]) for i in range(self.num_actors - 1)]
        # Add waypoints to the list for the ego vehicle
        self.waypoints.append(ego_waypoints[...,0:2])


        # Call reset on the prediction wrapper
        self.states, self.actions = self.prediction_wrapper.reset(initial_states, initial_actions)

        # Generate the output dictionary
        output = {
            "speeds" : self.states[..., 1:2],
            "steers" : self.states[..., 0:1],
            "poses" : self.poses,
            "waypoints" : self.waypoints
        }

        return output


    def step(self, actions, dones):
        """Step the actor manager forward by one step. Given the actions, sample the next state and observation.

        Args:
            actions: Torch tensor containing the actions of the actors at the past frame_stack steps
            dones: Torch tensor containing the done flags for the actors

        Returns:
            output (dict): Dictionary containing the following keys
                - speeds: Torch tensor containing the speed of the actors at the current frame_stack steps
                - steers: Torch tensor containing the steer of the actors at the current frame_stack steps
                - poses: Torch tensor containing the pose of the actors at the current frame_stack steps
        """

        ## Run forward pass on dynamics model to get updated state of all agents
        self.actions = actions

        #TODO: Don't do forward simulation for done actors

        # Call forward pass on prediction wrapper to get the updated states
        self.state, delta_poses = self.prediction_wrapper.step(actions)

        ## Compute updated pose for all vehicles
        # Compute the new vehicle pose for all vehicles
        # loc_delta contains the delta in position in the global coordinate frame
        # delta_poses[... ,0:2][...,None] reshapes the tensor to [num_actors, 2, 1]
        loc_delta = torch.squeeze(
                torch.bmm(
                    feutils.rot(torch.deg2rad(self.poses[:,2])),
                    delta_poses[... ,0:2][...,None]
                )
            )

        # Compute updated location and rotation
        loc = self.poses[:,0:2] + loc_delta
        rot = self.poses[:,  2] + delta_poses[...,2]
        rot = torch.unsqueeze(rot, dim = -1)

        # Update the pose of all the vehicles
        self.old_poses = self.poses
        self.poses = torch.cat([loc, rot], dim = -1)

        # Constrain theta to be between -180 and 180
        self.poses[self.poses[...,2] < -180][...,2] += 360
        self.poses[self.poses[...,2] >  180][...,2] -= 360

        # Generate the output dictionary
        output = {
            "speeds" : self.state[..., 1:2],
            "steers" : self.state[..., 0:1],
            "poses" : self.poses,
            "waypoints" : self.waypoints
        }

        return output

    def render(self, dones, waypoints):
        img = cv2.imread("/home/scratch/swapnilp/Town01.png")

        # Draw circles for each vehicle pose

        # Blue color in BGR
        color1 = (0, 255, 0)
        color2 = (0, 0, 255)
        color3 = (0, 255, 255)
        color4 = (255, 255, 0)
        color5 = (255, 0, 255)
        colors = [color1, color2, color3, color4, color5]
        color6 = (255, 255, 255)
        for i in range(self.poses.shape[0]):


            img_coordinate = self.world_to_pixel((self.poses[i,0], self.poses[i,1]))
            if (i < 1):
                cv2.circle(img, tuple(img_coordinate), 10, colors[0], 5)
            elif(i < 5):
                cv2.circle(img, tuple(img_coordinate), 10, colors[1], 5)
            elif dones[i]:
                # Draw rectangle cenetered at img_coordinate
                cv2.rectangle(img, (img_coordinate[0] - 10, img_coordinate[1] - 10),
                              (img_coordinate[0] + 10, img_coordinate[1] + 10), color6, -1)

        for wp_list in waypoints:
            for wp in wp_list:
                img_coordinate = self.world_to_pixel((wp[0], wp[1]))
                cv2.circle(img, tuple(img_coordinate), 10, color3, 5)
        img  = cv2.resize(img, (2000, 2000))
        cv2.imwrite(f"/home/scratch/swapnilp/test/{self.i:04d}.png", img)

        self.i += 1

    def world_to_pixel(self, location):
        x = SCALE * PIXELS_PER_METER * (location[0] - WORLD_OFFSET['Town01'][0])
        y = SCALE * PIXELS_PER_METER * (location[1] - WORLD_OFFSET['Town01'][1])
        return np.array([int(x), int(y)])






if __name__ == "__main__":
    # Imports here that are just needed for testing
    import projects.morel_mopo.algorithm.fake_envs.fake_env as fake_env
    from common.loggers.comet_logger import CometLogger
    from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig
    from projects.morel_mopo.config.morel_mopo_config import DefaultMLPObstaclesMOPOConfig
    from tqdm import tqdm
    import ipdb

    gpu = 0
    # Create a config
    config = DefaultMLPObstaclesMOPOConfig()
    config.populate_config(
        gpu = gpu,
        policy_algorithm = "PPO",
        pretrained_dynamics_model_key = "e1a27faf07f9450a87e6e6c10f29b0d8",
        pretrained_dynamics_model_name = "final"
    )

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = config.pretrained_dynamics_model_config.key
    temp_logger = CometLogger(logger_conf)

    # Load dynamics config
    temp_config = temp_logger.pickle_load("mopo", "config.pkl")
    dynamics_config = temp_config.dynamics_config
    dynamics_config.dataset_config.dataset_paths = [
                "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_dense_noisy_policy_traffic_lights",
                # "/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty_random_policy_new"
            ]
    print(f"MOPO: Loading dynamics model {config.pretrained_dynamics_model_config.name} from experiment {config.pretrained_dynamics_model_config.key}")

    dynamics = dynamics_config.dynamics_model_type.load(
                logger = temp_logger,
                model_name = config.pretrained_dynamics_model_config.name,
                gpu = gpu,
                data_config = dynamics_config.dataset_config)

    data_module = dynamics.data_module #self.dynamics_config.dataset_config.dataset_type(self.dynamics_config.dataset_config)

    fake_env_config = config.fake_env_config
    fake_env = dynamics_config.fake_env_type(dynamics,
                    config = fake_env_config,
                    logger = None)

    # Now we want to call the FakeEnv in a loop - policy actions don't actually matter here
    # Time the loop
    import time

    done = False
    obs = fake_env.reset()
    for i in range(100):

        actions = torch.zeros(obs.shape[0], 2)
        print('actions.shape:', actions.shape)
        start_time = time.time()
        obs, reward, done, _ = fake_env.step(actions)
        print('reward.shape:', reward.shape, 'done.shape:', done.shape)
        print(f"Time taken: {time.time() - start_time}")


### DEPRECATED

# class AutopilotPolicy(Policy):
#     """ A policy implementing our predefined autopilot policy. """

#     def __init__(self):
#         # Initialize the policy
#         super(AutopilotPolicy, self).__init__()

#         # Define the target speed for the autopilot
#         self.target_speed = 1.0

#         # ONLY FOR EXPERT AUTOPILOT
#         self.args_lateral_dict = {
#             'K_P': 0.88,
#             'K_D': 0.02,
#             'K_I': 0.5,
#             'dt': 1/10.0}

#     def reset(self, n_agents):
#         """Reset the policy.

#         Args:
#             n_agents: Number of agents in the environment
#         """

#         # Create n_agents number of controllers
#         self.lateral_controllers = [
#                                     feutils.PIDLateralController(
#                                         K_P=self.args_lateral_dict['K_P'],
#                                         K_D=self.args_lateral_dict['K_D'],
#                                         K_I=self.args_lateral_dict['K_I'],
#                                         dt=self.args_lateral_dict['dt'])

#                                     for _ in range(n_agents)
#                                 ]

#     def single_forward(self, i, observation):
#         """Generate action given the current state of the environment for a single agent.

#         Args:
#             observation: Torch tensor containing the current observation for the policy

#         Returns:
#             Torch tensor containing the action to take
#         """
#         # This is done, return 0, 1
#         if observation[-1]:
#             return torch.Tensor([0,-1])

#         ## Split the observation into the relevant components
#         # Retrieve next waypoint
#         waypoint = observation[0:2]

#         # (x,y, theta)
#         vehicle_pose = observation[2:5]

#         # Obstacle distance
#         obstacle_dist = observation[5]

#         # Compute steering angle
#         steer = self.lateral_controllers[i].pid_control(vehicle_pose.cpu().numpy(), waypoint)
#         steer = np.clip(steer, -1, 1)

#         # Compute speed
#         target_speed = self.target_speed
#         # Stop if the obstacle is too close
#         if(obstacle_dist < 0.3):
#             target_speed = -1.0

#         return torch.Tensor([steer, target_speed])

#     def forward(self, observations):
#         """Generate the action given the current state of the environment.

#         Args:
#             observations: Torch tensor containing the current observation for the policy

#         Returns:
#             Torch tensor containing the action to take
#         """

#         # Generate the action for each agent
#         # Strange code to handle scope within the list comprehension
#         actions = torch.stack(
#             (lambda single_forward = self.single_forward, observations=observations:
#                 [single_forward(i, observation) for i, observation in enumerate(observations)]
#             )()
#         )

#         return actions




# class Policy:
#     """Base class to define the policy API.

#     A policy is responsible for generating the action given it's associated observation. The policy is defined separately from the actor so that we can batch computation for the actions
#     for a set of homogeneous actors.
#     """

#     def __init__(self):
#         """Initialize the policy."""

#     def reset(self, n_agents):
#         """Reset the policy to the initial state at the start of the new episode.

#         Args:
#             n_agents: Number of agents in the environment.

#         Returns:
#             None
#         """

#         raise NotImplementedError

#     def forward(self, observations):
#         """Generate the action given the current state of the environment.

#         Args:
#             state: Torch tensor containing the current state of the environment

#         Returns:
#             Torch tensor containing the action to take
#         """

#         raise NotImplementedError


#     def __call__(self, observations):
#         """Generate the action given the current state of the environment.

#         Args:
#             state: Torch tensor containing the current state of the environment

#         Returns:
#             Torch tensor containing the action to take
#         """

#         return self.forward(observations)

# class NPCActorGroup():
#     """ Class to handle all actors of the same group

#     Completes the instantiation, resetting, and stepping of all homogeneous actors
#     """

#     def __init__(self, actor_type, config):
#         """Initialize the actor group

#         Args:
#             actor_type: Type of actor to instantiate
#             actor_states: Torch tensor containing the states of the actor for the entire rollout
#             actor_indices: List of indices of the actors in the rollout
#             config: Config object containing the configuration for the actor
#         """

#         # Create a policy for the actors
#         self.policy = actor_type.POLICY_TYPE()
#         self.actor_type = actor_type
#         self.config = config

#     def set_device(self, device):
#         self.device = device

#     def step(self, new_states, new_poses):
#         """Step the actor manager forward by one step.

#         Args:
#             state: Torch tensor containing the states of the actors at the current step
#         """

#         # Retrieve new observations the policy from each actor
#         # start = time.time()
#         all_observations = [actor.update_state(new_states, new_poses) for actor in self.actors]
#         # end = time.time()
#         print("Time taken to update state : ", end - start)
#         # Pass stack of observations to the policy to compute all actions
#         all_actions = self.policy(all_observations)

#         # Update the actions in the actors
#         map(lambda actor, action: actor.update_action(action), self.actors, all_actions)

#         # Return the actions for forward rollout of the dynamics
#         return torch.atleast_2d(all_actions)


# class Actor:
#     """Base class to define the actor API

#     An actor is responsible for maintaining its current state and generating its own observations given the environment state. The actor does NOT actually perform the forward pass to compute its actions or the dynamic rollout.
#     """

#     def __init__(self, all_actor_trajectories, actor_idx, config):
#         """Initialize the actor.

#         Args:
#             actor_states: Torch tensor containing the states of the actor for the entire rollout
#         """

#         raise NotImplementedError

#     def update_state(self, state):
#         """Update the actor state given the new environment state.

#         Args:
#             state: Torch tensor containing the new environment state

#         Returns:
#             Torch tensor containing the new observation for the policy
#         """

#         raise NotImplementedError


#     def update_action(self, action):
#         """Store the action that the actor has taken.

#         Args:
#             action: Torch tensor containing the new action of the actor

#         Returns:
#             None
#         """

#         raise NotImplementedError

# class AutopilotActor(Actor):
#     """ Actor that runs our predefined autopilot policy. """

#     POLICY_TYPE = AutopilotPolicy

#     def __init__(self, all_actor_trajectories, actor_idx, config, device):
#         """Initialize the actor.

#         Args:
#             all_actor_trajectories: Torch tensor containing the trajectories for all actors in the rollout [num_actors, trajectory_length, 4]
#                           [:,:,0]: x
#                           [:,:,1]: y
#                           [:,:,2]: theta
#                           [:,:,3]: v
#             actor_idx: Index of the actor in the rollout
#         """
#         # Init super class
#         # super(AutopilotActor, self).__init__(all_actor_trajectories, actor_idx, config)

#         # Save important parameters here
#         self.config = config
#         self.device = device

#         # Save the index of the actor
#         self.actor_idx = actor_idx

#         # Convert trajectory of actor states into a trajectory of waypoints
#         try:
#             self.waypoints = feutils.get_waypoints(all_actor_trajectories[:, actor_idx, :])
#         except:
#             import ipdb; ipdb.set_trace()
#         # Get the vehicle proximity distance from the config
#         self.vehicle_proximity_threshold = self.config.obs_config.vehicle_proximity_threshold

#         # Save the initial poses of all of the other actors
#         # NOTE: This contains the state of self as well currently
#         self.other_actor_poses = all_actor_trajectories[0, :, 0:3]

#         # Stores whether the actor rollout is finished
#         self.done = False

#     def update_state(self, new_states, new_poses):
#         """Update the actor state given the new environment state.

#         Args:
#             all_vehicle_states: torch tensor containing the new state dynamics state of the actors
#                     [num_actors, 2] [speed,steer]
#             all_vehicle_poses: torch tensor containing the new pose of all actors
#                     [num_actors, 3] [x, y, theta]
#         Returns:
#             Torch tensor containing the new observation for the policy
#         """

#         # Update the actor state
#         try:
#             self.cur_pose = new_poses[self.actor_idx]
#             self.cur_velocity = new_states[self.actor_idx, 0]
#         except:
#             import ipdb; ipdb.set_trace()
#         # NOTE: This contains the state of self as well
#         self.other_actor_poses = new_poses
#         # Return the new observation
#         return self.get_observation()

#     def get_observation(self):
#         """Get the observation from the current state.

#         Returns:
#             Torch tensor containing the observation
#         """

#         #### Observation for autopilot policy
#         # observation[0:2]: Next waypoint
#         # observation[2:5]: Pose of the vehicle (x, y, theta)
#         # observation[5]: Distance to the obstacle ahead of our vehicle

#         # Process waypoints
#         start = time.time()
#         angle, \
#         dist_to_trajectory, \
#         remaining_waypoints, \
#         self.second_last_waypoint, \
#         self.last_waypoint, \
#         self.previous_waypoint = feutils.process_waypoints(self.waypoints,
#                                                         self.cur_pose,
#                                                         self.device)
#         self.waypoints = remaining_waypoints

#         # next_waypoints,\
#         # _, _, \

#         print( "Time to process waypoints: {}".format(time.time() - start))

#         if(len(self.waypoints) == 1):
#             self.done = True

#         start = time.time()
#         if(not self.done):
#             # Retrieve next waypoint
#             if(len(self.waypoints) >= 3):
#                 waypoint = self.waypoints[2]
#             else:
#                 waypoint = self.waypoints[-1]
#             # (x,y, theta)
#             vehicle_pose = self.cur_pose


#             # Obstacle distance
#             obstacle_dist = 1.0
#             for i in range(self.other_actor_poses.shape[0]):
#                 # Skip if the i is the current actor
#                 if(i == self.actor_idx):
#                     continue

#                 d_bool, norm_target = feutils.is_within_distance_ahead(
#                                         self.other_actor_poses[i],
#                                         self.cur_pose,
#                                         self.vehicle_proximity_threshold
#                                     )

#                 if(d_bool):
#                     obstacle_dist = norm_target/self.config.obs_config.vehicle_proximity_threshold
#                     break

#             print( "Time to process obstacle distance: {}".format(time.time() - start))

#             # Return the observation
#             return torch.Tensor([waypoint[0], waypoint[1], vehicle_pose[0], vehicle_pose[1], vehicle_pose[2], obstacle_dist, self.done])
#         return torch.Tensor([0, 0, 0, 0, 0, 0, self.done])

#     def update_action(self, action):
#         """Store the action that the actor has taken.

#         Args:
#             action: Torch tensor containing the new action of the actor

#         Returns:
#             None
#         """

#         # Store the action taken
#         self.action = action
