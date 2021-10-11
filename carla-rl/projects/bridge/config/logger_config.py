# Use Base Config from environment to build config
from common.loggers.comet_logger_config import BaseCometLoggerConfig



class CometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "roUfs00hpDDbntO56E1FrQ29b"
        self.workspace = "swapnilpande"
        self.project_name = "bridge"
        self.log_dir = "/home/scratch/swapnilp/bridge"
        self.pdb_on_exception = False


class ExistingCometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "roUfs00hpDDbntO56E1FrQ29b"
        self.experiment_key = None
        self.log_dir = "/home/scratch/swapnilp/bridge"
        self.pdb_on_exception = True
