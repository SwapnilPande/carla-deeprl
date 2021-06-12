# Use Base Config from environment to build config
from common.loggers.comet_logger_config import BaseCometLoggerConfig




class CometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "Vvk4XNATcbTMmB9UraFl7IQzR"
        self.workspace = "swapnilpande"
        self.project_name = "morel-carla"
        self.log_dir = "/home/scratch/vccheng/carla-rl_testing"

class ExistingCometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "Vvk4XNATcbTMmB9UraFl7IQzR"
        self.experiment_key = None
        self.log_dir = "/home/scratch/vccheng/carla-rl_testing"