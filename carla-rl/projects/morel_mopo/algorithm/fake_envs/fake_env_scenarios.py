import numpy as np
import torch
from projects.morel_mopo.algorithm.fake_envs import fake_env_utils as feutils


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
        self.waypoints = torch.zeros(30, 3)
        self.waypoints[:, 0] = torch.linspace(0, 50, 30)

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
        vehicle_pose = torch.tensor([10.0, 0.0, 0.0])


        out = (self.obs.normalized, self.action.normalized, None, None, None, vehicle_pose)

        return out, self.waypoints, [self.obstacle_pose] * timeout_steps

class TurnWithObstacle():

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
        self.waypoints = torch.zeros(42,3)
        self.waypoints[:20, 0] = torch.linspace(0, 45, 20)
        self.waypoints[20, 0] =  torch.tensor([46.5])
        self.waypoints[20, 1] = torch.tensor([1.5])
        self.waypoints[21, 0] = torch.tensor([48.0])
        self.waypoints[21, 1] = torch.tensor([3.0])
        self.waypoints[22:, 0] = 50
        self.waypoints[22:, 1] = torch.linspace(5, 50, 20)

        #import ipdb
        #ipdb.set_trace()
        


         # Create a static obstacle at 40 meters down the road
        # Dimensions are (x,y,theta, speed)

        self.obstacle_pose = torch.tensor([[50.0, 5.0, 0.0, 0.0]])
        

        # Obs contains [steer, speed] at every past frame
        # Set all of these to zero
        self.obs.unnormalized = torch.zeros(self.frame_stack, 2)

        # Action contains the [steer_cmd, speed_cmd] at every past frame
        # Set all of these to zero as well
        self.action.unnormalized = torch.zeros(self.frame_stack, 2)

       
    def sample_with_waypoints(self, timeout_steps):
        # Timeout steps specify how long the list of obstacle_pose should be

        # Initialize vehicle facing forward at random distance before obstacle
        vehicle_pose = torch.tensor([10.0, 0.0, 0.0])

        #print('Sampling from TurnWithObstacle')


        out = (self.obs.normalized, self.action.normalized, None, None, None, vehicle_pose)

        return out, self.waypoints, [self.obstacle_pose] * timeout_steps



if __name__ == "__main__":
    import ipdb; ipdb.set_trace()