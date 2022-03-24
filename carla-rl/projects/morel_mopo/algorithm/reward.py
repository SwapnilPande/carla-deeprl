import torch

'''
Calculates reward according to distance to trajectory
@params next_state:   vehicle state after taking action
        dist_to_traj: distance from vehicle to current trajectory
        config
'''
def compute_reward(speeds, dist_to_trajectories, collisions, config):

    speeds = speeds.squeeze()
    dist_to_trajectories = dist_to_trajectories.squeeze()
    collisions = collisions.squeeze()


    reward_config = config.reward_config

    # Speed reward
    speed = speeds * reward_config.speed_coeff
    # Dist to trajectory
    dist_to_trajectory = dist_to_trajectories * reward_config.dist_to_trajectory_coeff

    collision_penalty = reward_config.const_collision_penalty * collisions
#     collision_penalty += 10 * next_state[1]

    # Penalize for straying from target trajectory
    #off_route = torch.abs(dist_to_trajectory) > 10
    return speed - torch.abs(dist_to_trajectory) * reward_config.dist_to_trajectory_coeff - collision_penalty
