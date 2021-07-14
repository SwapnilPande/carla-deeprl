from environment.config.base_config import BaseConfig


class BaseUncertaintyConfig(BaseConfig):
    def __init__(self):
        super().__init__()

        self.uncertainty_coeff = None
        # this value, multiplied by the uncertainty, penalized in the reward function
        self.uncertainty_penalty_coeff = None

 

class DefaultUncertaintyConfig(BaseUncertaintyConfig):
    def __init__(self):

        self.uncertainty_coeff = 100

        self.uncertainty_penalty_coeff = 150