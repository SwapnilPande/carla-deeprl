from environment.config.base_config import BaseConfig



class BaseRewardConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        # Speed reward coefficient
        self.speed_coeff = None

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = None

        # Multiply this by dist to trajectory as penalty
        self.dist_to_trajectory_penalty_coeff = None

        # Penalize if goes off-route
        self.off_route_penalty_coeff = None

 

class DefaultRewardConfig(BaseRewardConfig):
    def __init__(self):
        # Speed reward coefficient
        self.speed_coeff = 15

        # Coefficient for dist_to_trajec reward
        # Pass a positive value for this argument
        self.dist_to_trajectory_coeff = 10

        # subtract speed by this * dist_to_trajectory 
        self.dist_to_trajectory_penalty_coeff = 1.2

        # subtract speed by this * off_route
        self.off_route_penalty_coeff = 50.0
