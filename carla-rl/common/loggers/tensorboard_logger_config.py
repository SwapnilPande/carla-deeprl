# Use Base Config from environment to build config
from common.loggers.base_logger_config import BaseLoggerConfig


class BaseTensorboardLoggerConfig(BaseLoggerConfig):
    def __init__(self):
        super().__init__()
        # Existing experiment key
        # This should only be passed if loading existing experiment
        self.experiment_key = None

        ##### These fields should be populated every experiment
        # Name of the experiment
        # If name already exists in logging directory
        self.experiment_name = None

        # Directory in which to store logs
        self.log_dir = None

    def populate(self, experiment_name):
        self.experiment_name = experiment_name
        self.tags = []

    def verify(self):
        # None of these need to be passed if we passed experiment key
        if(self.experiment_key is not None):
            super().verify(ignore_keys = ["tags", "experiment_name"])
        # If none, all other fields must be populated
        else:
            super().verify(ignore_keys = ["experiment_key"])