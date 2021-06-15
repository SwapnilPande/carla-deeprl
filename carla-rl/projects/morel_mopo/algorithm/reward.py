import torch 

''' 
Calculates reward according to distance to trajectory 
@params next_state:   vehicle state after taking action
        dist_to_traj: distance from vehicle to current trajectory
        config
'''
def compute_reward(next_state, dist_to_traj, config):
    
    reward_config = config.reward_config

    # Speed reward
    speed = next_state[0] * reward_config.speed_coeff
    # Dist to trajectory 
    dist_to_trajectory = dist_to_traj * reward_config.dist_to_trajectory_coeff

    # Penalize for straying from target trajectory 
    off_route = torch.abs(dist_to_trajectory) > 10
    return torch.unsqueeze(speed - (torch.abs(dist_to_trajectory) * reward_config.dist_to_trajectory_penalty_coeff) \
                            - (off_route * reward_config.off_route_penalty_coeff), dim = 0)
