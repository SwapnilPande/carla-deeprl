import numpy as np
import torch
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils
from copy import deepcopy

class StraightWithStaticObstacle():

    def __init__(self, frame_stack, norm_stats):

        # Number of frames to stack the initial state for
        self.frame_stack = frame_stack

        # Save the statistic for normalizing the data correctly
        self.norm_stats = norm_stats


        # Build tensors for storing data
        self.obs = feutils.NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], "cpu")
        self.action = feutils.NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], "cpu")

        # Waypoints are a series of waypoints 5 m apart
        # These waypoints denote a straight route down a road
        # Contains a total of 11 waypoints, each of which are 5 meters apart
        self.waypoints = torch.zeros(11, 3)
        self.waypoints[:, 0] = torch.linspace(0, 50, 11)

        # Obs contains [steer, speed] at every past frame
        # Set all of these to zero
        self.obs.unnormalized = torch.zeros(self.frame_stack, 2)

        # Action contains the [steer_cmd, speed_cmd] at every past frame
        # Set all of these to zero as well
        self.action.unnormalized = torch.zeros(self.frame_stack, 2)

        # Create a static obstacle at 40 meters down the road
        # Dimensions are (x,y,theta, speed)
        self.obstacle_pose = torch.tensor([[40.0, 0.0, 0.0, 0.0]])


    def sample_with_waypoints(self, timeout_steps):
        # Timeout steps specify how long the list of obstacle_pose should be

        # Initialize vehicle facing forward at random distance before obstacle
        vehicle_pose = torch.tensor([np.random.rand() * 35, 0.0, 0.0])


        out = (self.obs.normalized, self.action.normalized, None, None, None, vehicle_pose)

        return out, self.waypoints, [self.obstacle_pose] * timeout_steps



class StraightWithObstacle():

    def __init__(self, frame_stack, norm_stats):

        # Number of frames to stack the initial state for
        self.frame_stack = frame_stack

        # Save the statistic for normalizing the data correctly
        self.norm_stats = norm_stats


        # Build tensors for storing data
        self.obs = feutils.NormalizedTensor(self.norm_stats["obs"]["mean"], self.norm_stats["obs"]["std"], "cpu")
        self.action = feutils.NormalizedTensor(self.norm_stats["action"]["mean"], self.norm_stats["action"]["std"], "cpu")

        # Waypoints are a series of waypoints 5 m apart
        # These waypoints denote a straight route down a road
        # Contains a total of 11 waypoints, each of which are 5 meters apart
        num_waypoints = 90//2
        self.waypoints = torch.zeros(num_waypoints, 3)
        self.waypoints[:, 0] = torch.linspace(0, 90, num_waypoints)

        # Obs contains [steer, speed] at every past frame
        # Set all of these to zero
        self.obs.unnormalized = torch.zeros(self.frame_stack, 2)

        # Action contains the [steer_cmd, speed_cmd] at every past frame
        # Set all of these to zero as well
        self.action.unnormalized = torch.zeros(self.frame_stack, 2)

        # Create a static obstacle at 40 meters down the road
        # Dimensions are (x,y,theta, speed)
        self.obstacle_pose = torch.tensor([[40.0, 0.0, 0.0, 0.0]])


    def sample_with_waypoints(self, timeout_steps):
        # Timeout steps specify how long the list of obstacle_pose should be

        # Initialize vehicle facing forward at random distance before obstacle
        vehicle_pose = torch.tensor([32, 0.0, 0.0])

        # Remove all waypoints that are less than the vehicle pose
        waypoints_to_use = self.waypoints[self.waypoints[:, 0] > vehicle_pose[0]]


        out = (self.obs.normalized, self.action.normalized, None, None, None, vehicle_pose)

        x = 45.0
        obstacle_poses = [deepcopy(self.obstacle_pose)]
        for i in range(5):
            obstacle_poses.append([[x, 0.0, 0.0, 0.0]])

            x += 5

        for i in range(timeout_steps - 10):
            obstacle_poses.append([[x, 0.0, 0.0, 0.0]])

        return out, waypoints_to_use, obstacle_poses






if __name__ == "__main__":
    import ipdb; ipdb.set_trace()