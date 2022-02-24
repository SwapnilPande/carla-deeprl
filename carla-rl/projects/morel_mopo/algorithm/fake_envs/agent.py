from collections import OrderedDict
import projects.morel_mopo.algorithm.fake_envs.fake_env_utils as feutils

import numpy as np
import torch
import time

class Policy:
    """Base class to define the policy API.

    A policy is responsible for generating the action given it's associated observation. The policy is defined separately from the actor so that we can batch computation for the actions
    for a set of homogeneous actors.
    """

    def __init__(self):
        """Initialize the policy."""

    def reset(self, n_agents):
        """Reset the policy to the initial state at the start of the new episode.

        Args:
            n_agents: Number of agents in the environment.

        Returns:
            None
        """

        raise NotImplementedError

    def forward(self, observations):
        """Generate the action given the current state of the environment.

        Args:
            state: Torch tensor containing the current state of the environment

        Returns:
            Torch tensor containing the action to take
        """

        raise NotImplementedError


    def __call__(self, observations):
        """Generate the action given the current state of the environment.

        Args:
            state: Torch tensor containing the current state of the environment

        Returns:
            Torch tensor containing the action to take
        """

        return self.forward(observations)


class AutopilotPolicy(Policy):
    """ A policy implementing our predefined autopilot policy. """

    def __init__(self):
        # Initialize the policy
        super(AutopilotPolicy, self).__init__()

        # Define the target speed for the autopilot
        self.target_speed = 1.0

        # ONLY FOR EXPERT AUTOPILOT
        self.args_lateral_dict = {
            'K_P': 0.88,
            'K_D': 0.02,
            'K_I': 0.5,
            'dt': 1/10.0}

    def reset(self, n_agents):
        """Reset the policy.

        Args:
            n_agents: Number of agents in the environment
        """

        # Create n_agents number of controllers
        self.lateral_controllers = [
                                    feutils.PIDLateralController(
                                        K_P=self.args_lateral_dict['K_P'],
                                        K_D=self.args_lateral_dict['K_D'],
                                        K_I=self.args_lateral_dict['K_I'],
                                        dt=self.args_lateral_dict['dt'])

                                    for _ in range(n_agents)
                                ]

    def single_forward(self, i, observation):
        """Generate action given the current state of the environment for a single agent.

        Args:
            observation: Torch tensor containing the current observation for the policy

        Returns:
            Torch tensor containing the action to take
        """
        # This is done, return 0, 1
        if observation[-1]:
            return torch.Tensor([0,-1])

        ## Split the observation into the relevant components
        # Retrieve next waypoint
        waypoint = observation[0:2]

        # (x,y, theta)
        vehicle_pose = observation[2:5]

        # Obstacle distance
        obstacle_dist = observation[5]

        # Compute steering angle
        steer = self.lateral_controllers[i].pid_control(vehicle_pose.cpu().numpy(), waypoint)
        steer = np.clip(steer, -1, 1)

        # Compute speed
        target_speed = self.target_speed
        # Stop if the obstacle is too close
        if(obstacle_dist < 0.3):
            target_speed = -1.0

        return torch.Tensor([steer, target_speed])

    def forward(self, observations):
        """Generate the action given the current state of the environment.

        Args:
            observations: Torch tensor containing the current observation for the policy

        Returns:
            Torch tensor containing the action to take
        """

        # Generate the action for each agent
        # Strange code to handle scope within the list comprehension
        actions = torch.stack(
            (lambda single_forward = self.single_forward, observations=observations:
                [single_forward(i, observation) for i, observation in enumerate(observations)]
            )()
        )

        return actions





class ActorManager():
    """Class to handle the heteregeneous set of actors that we need to maintain the base_fake env.

    This class is designed to batch computation as much as possible for rolling out the various actors in the environment. The actor manager will also maintain the buffer of available policies to sample from.
    """

    def __init__(self, config, frame_stack, norm_stats, new_first, device):

        self.config = config
        self.frame_stack = frame_stack
        self.device = device
        self.norm_stats = norm_stats
        self.new_first = new_first

        # Dictionary to store the various actors to roll out
        # Each entry in the list is a HomogeneousActorGroup object that handles all actors of the same type
        self.actor_groups = []

        # Sampling probabilities for each type of actor
        self.actor_type_probabilities = []

        # Create tensors to store the history of the states and actions of all actors
        # We use normalized tensors to naturally handle normalized/unnormalized data
        self.state = feutils.NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], self.device)
        self.past_action = feutils.NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], self.device)

        # num_actors stores the total number of actors
        self.num_actors = 0


    def add_actor_group(self, actor_group, sampling_probability):
        """ Add a homogeneous actor group to sample from. The actor group contains all actors of a specific type.

        Args:
            actor_group: Instance of HomogeneousActorGroup class
            sampling_probability: Probability of sampling this actor type
        """

        self.actor_groups.append(actor_group)
        self.actor_groups[-1].set_device(self.device)
        self.actor_type_probabilities.append(sampling_probability)
        self.actor_probabilities = [float(i)/sum(self.actor_type_probabilities) for i in self.actor_type_probabilities]

    def reset(self, actor_trajectories):
        """Reset the actor manager to the initial state of the new episode.

        Args:
            actor_trajectories: Torch tensor containing the states of the pose and velocity
                                 for the entire rollout
                                [time_steps, num_actors, 4 (x,y,theta,speed)]
        """

        self.num_actors = actor_trajectories.shape[1]

        # Sample the actor types to use
        # Right now, just get the indices of the relevant actor types
        sampled_actor_types = np.random.choice(len(self.actor_groups),
                                                size = self.num_actors,
                                                p=self.actor_type_probabilities,
                                                replace = True)

        # Given actors types for each actor, reorder actor_states
        # such that the actors of the same type are grouped together
        # This is done to make batching computations easier

        # Create new array of same shape as actor_states
        sorted_actor_trajectories = torch.zeros_like(actor_trajectories)
        # Array to store the indices of the actors of each type
        actor_idx_ranges = torch.zeros(len(self.actor_groups), dtype=torch.int)

        # For each actor type, copy the relevant actors to the new array
        cur_idx = 0
        for i, actor_group in enumerate(self.actor_groups):
            # Get the indices of the actors of this type
            actor_indices = np.where(sampled_actor_types == i)[0]

            # num_actors_of_type is the number of actors of this type
            num_actors_of_type = len(actor_indices)

            # Copy the relevant actors to the new array
            actor_idx_ranges[i] = cur_idx + num_actors_of_type

            sorted_actor_trajectories[:, cur_idx:actor_idx_ranges[i], :] = actor_trajectories[:, actor_indices, :]

        actor_trajectories = sorted_actor_trajectories

        # Reset for each actor group
        prev_actor_idx = 0
        for i, actor_group in enumerate(self.actor_groups):
            # Pass all of the actor states, as well as the sampled actor indices
            actor_group.reset(actor_trajectories, np.arange(prev_actor_idx, actor_idx_ranges[i]))
            prev_actor_idx = actor_idx_ranges[i]

        # State input : [frame_stack, 2]
        #       Speed
        #       Steer
        # We don't actually have these values for autopilot actors
        # Dumb solution to this is to initialize actors to 0 speed and steer
        # TODO : Figure out a better solution
        # Reset the state of all of the actors
        self.state.unnormalized = torch.zeros(self.num_actors,
                                              self.frame_stack,
                                              2,
                                              device=self.device)

        # Save the initial pose of all the actors
        # The initial pose of the actors is simply the first time step in the trajectories
        # Don't include the 4 element for each actor, as this is the actor velocity, which we are ignoring
        # and setting to 0
        self.poses = actor_trajectories[0, :, 0:3]

        # If the current speed/steer is 0, then the actions should also be 0, this is actually -1 for the target_speed
        self.past_action.unnormalized = torch.zeros(self.num_actors,
                                                    self.frame_stack,
                                                    2,
                                                    device=self.device)
        # Set the target_speed to -1 for all actors - this corresponds to full braking
        self.past_action.unnormalized[...,1] = -1

        # After finishing the reset, we need to call step to get the actions of all of the actors
        # Call step on our current state to get the actions
        return self.step(self.state, self.poses, self.past_action)

    def step(self, states, poses, actions):
        """Step the actor manager forward by one step.

        Args:
            state: Torch tensor containing the states of the actors at the past frame_stack steps
            poses: Torch tensor containing the poses of the actors at hte current step
            actions: Torch tensor containing the actions of the actors at the past frame_stack steps
        """

        # Store the updated state and actions returned from the fake_env
        self.state = states
        self.poses = poses
        self.past_actions = actions

        # Retrieve new state and action
        new_states = self.state.unnormalized[:, 0, :] if self.new_first else self.state.unnormalized[:, -1, :]

        # Retrieve new actions from the actors
        cur_idx = 0
        new_actions = torch.empty(self.num_actors, 2, device=self.device)
        for i, actor_group in enumerate(self.actor_groups):
            # Get new actions
            new_actor_group_actions = actor_group.step(new_states, self.poses)


            # Copy the new actions to the new action array
            new_actions[cur_idx:cur_idx+new_actor_group_actions.shape[0]] = new_actor_group_actions

            # Update the current index
            cur_idx += new_actor_group_actions.shape[0]



        # The fake env requires the state, past_action, and new_actions
        # We return the state and action as NormalizedTensor objects
        return self.state, self.poses, self.past_action, new_actions



class HomogeneousActorGroup():
    """ Class to handle all actors of the same group

    Completes the instantiation, resetting, and stepping of all homogeneous actors
    """

    def __init__(self, actor_type, config):
        """Initialize the actor group

        Args:
            actor_type: Type of actor to instantiate
            actor_states: Torch tensor containing the states of the actor for the entire rollout
            actor_indices: List of indices of the actors in the rollout
            config: Config object containing the configuration for the actor
        """

        # Create a policy for the actors
        self.policy = actor_type.POLICY_TYPE()
        self.actor_type = actor_type
        self.config = config

    def set_device(self, device):
        self.device = device

    def reset(self, actor_trajectories, actor_indices):
        """Reset the actor to the initial state of the new episode.

        Args:
            all_actor_trajectories: Torch tensor containing the states of the actors for the entire rollout
            actor_indices: List of indices of the actors in the rollout
        """

        # Store the actor indices
        self.actor_indices = actor_indices

        # Instantiate all actors
        self.actors = [
            self.actor_type(actor_trajectories, actor_index, self.config, self.device) for actor_index in actor_indices
        ]

        self.num_actors = len(self.actors)

        # Next, we need to reset the policy
        # Policies will take the number of actors as an input
        self.policy.reset(self.num_actors)


    def step(self, new_states, new_poses):
        """Step the actor manager forward by one step.

        Args:
            state: Torch tensor containing the states of the actors at the current step
        """

        # Retrieve new observations the policy from each actor
        # start = time.time()
        all_observations = [actor.update_state(new_states, new_poses) for actor in self.actors]
        # end = time.time()
        print("Time taken to update state : ", end - start)
        # Pass stack of observations to the policy to compute all actions
        all_actions = self.policy(all_observations)

        # Update the actions in the actors
        map(lambda actor, action: actor.update_action(action), self.actors, all_actions)

        # Return the actions for forward rollout of the dynamics
        return torch.atleast_2d(all_actions)


class Actor:
    """Base class to define the actor API

    An actor is responsible for maintaining its current state and generating its own observations given the environment state. The actor does NOT actually perform the forward pass to compute its actions or the dynamic rollout.
    """

    def __init__(self, all_actor_trajectories, actor_idx, config):
        """Initialize the actor.

        Args:
            actor_states: Torch tensor containing the states of the actor for the entire rollout
        """

        raise NotImplementedError

    def update_state(self, state):
        """Update the actor state given the new environment state.

        Args:
            state: Torch tensor containing the new environment state

        Returns:
            Torch tensor containing the new observation for the policy
        """

        raise NotImplementedError


    def update_action(self, action):
        """Store the action that the actor has taken.

        Args:
            action: Torch tensor containing the new action of the actor

        Returns:
            None
        """

        raise NotImplementedError

class AutopilotActor(Actor):
    """ Actor that runs our predefined autopilot policy. """

    POLICY_TYPE = AutopilotPolicy

    def __init__(self, all_actor_trajectories, actor_idx, config, device):
        """Initialize the actor.

        Args:
            all_actor_trajectories: Torch tensor containing the trajectories for all actors in the rollout [num_actors, trajectory_length, 4]
                          [:,:,0]: x
                          [:,:,1]: y
                          [:,:,2]: theta
                          [:,:,3]: v
            actor_idx: Index of the actor in the rollout
        """
        # Init super class
        # super(AutopilotActor, self).__init__(all_actor_trajectories, actor_idx, config)

        # Save important parameters here
        self.config = config
        self.device = device

        # Save the index of the actor
        self.actor_idx = actor_idx

        # Convert trajectory of actor states into a trajectory of waypoints
        try:
            self.waypoints = feutils.get_waypoints(all_actor_trajectories[:, actor_idx, :])
        except:
            import ipdb; ipdb.set_trace()
        # Get the vehicle proximity distance from the config
        self.vehicle_proximity_threshold = self.config.obs_config.vehicle_proximity_threshold

        # Save the initial poses of all of the other actors
        # NOTE: This contains the state of self as well currently
        self.other_actor_poses = all_actor_trajectories[0, :, 0:3]

        # Stores whether the actor rollout is finished
        self.done = False

    def update_state(self, new_states, new_poses):
        """Update the actor state given the new environment state.

        Args:
            all_vehicle_states: torch tensor containing the new state dynamics state of the actors
                    [num_actors, 2] [speed,steer]
            all_vehicle_poses: torch tensor containing the new pose of all actors
                    [num_actors, 3] [x, y, theta]
        Returns:
            Torch tensor containing the new observation for the policy
        """

        # Update the actor state
        try:
            self.cur_pose = new_poses[self.actor_idx]
            self.cur_velocity = new_states[self.actor_idx, 0]
        except:
            import ipdb; ipdb.set_trace()
        # NOTE: This contains the state of self as well
        self.other_actor_poses = new_poses
        # Return the new observation
        return self.get_observation()

    def get_observation(self):
        """Get the observation from the current state.

        Returns:
            Torch tensor containing the observation
        """

        #### Observation for autopilot policy
        # observation[0:2]: Next waypoint
        # observation[2:5]: Pose of the vehicle (x, y, theta)
        # observation[5]: Distance to the obstacle ahead of our vehicle

        # Process waypoints
        start = time.time()
        angle, \
        dist_to_trajectory, \
        remaining_waypoints, \
        self.second_last_waypoint, \
        self.last_waypoint, \
        self.previous_waypoint = feutils.process_waypoints(self.waypoints,
                                                        self.cur_pose,
                                                        self.device)
        self.waypoints = remaining_waypoints

        # next_waypoints,\
        # _, _, \

        print( "Time to process waypoints: {}".format(time.time() - start))

        if(len(self.waypoints) == 1):
            self.done = True

        start = time.time()
        if(not self.done):
            # Retrieve next waypoint
            if(len(self.waypoints) >= 3):
                waypoint = self.waypoints[2]
            else:
                waypoint = self.waypoints[-1]
            # (x,y, theta)
            vehicle_pose = self.cur_pose


            # Obstacle distance
            obstacle_dist = 1.0
            for i in range(self.other_actor_poses.shape[0]):
                # Skip if the i is the current actor
                if(i == self.actor_idx):
                    continue

                d_bool, norm_target = feutils.is_within_distance_ahead(
                                        self.other_actor_poses[i],
                                        self.cur_pose,
                                        self.vehicle_proximity_threshold
                                    )

                if(d_bool):
                    obstacle_dist = norm_target/self.config.obs_config.vehicle_proximity_threshold
                    break

            print( "Time to process obstacle distance: {}".format(time.time() - start))

            # Return the observation
            return torch.Tensor([waypoint[0], waypoint[1], vehicle_pose[0], vehicle_pose[1], vehicle_pose[2], obstacle_dist, self.done])
        return torch.Tensor([0, 0, 0, 0, 0, 0, self.done])

    def update_action(self, action):
        """Store the action that the actor has taken.

        Args:
            action: Torch tensor containing the new action of the actor

        Returns:
            None
        """

        # Store the action taken
        self.action = action




if __name__ == "__main__":
    # Imports here that are just needed for testing
    import projects.morel_mopo.algorithm.fake_envs.fake_env as fake_env
    from common.loggers.comet_logger import CometLogger
    from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig
    from projects.morel_mopo.config.morel_mopo_config import DefaultMLPObstaclesMOPOConfig
    from tqdm import tqdm

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

    done = False
    obs = fake_env.reset()
    for i in range(100):
        actions = torch.Tensor([[0.0, 0.0]])
        obs, _, done, _ = fake_env.step(actions)



