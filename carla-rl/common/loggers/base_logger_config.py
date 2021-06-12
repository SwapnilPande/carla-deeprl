# Use Base Config from environment to build config
from environment.config.base_config import BaseConfig



class BaseLoggerConfig(BaseConfig):
    def __init__(self):
        ##### These fields should be populated every experiment
        # Name of the experiment
        # If name already exists in logging directory
        self.experiment_name = None

        # A list of tags to tag the experiment with
        self.tags = None

        # Directory in which to store logs
        self.log_dir = None

    def populate(self, experiment_name, tags = []):
        self.experiment_name = experiment_name
        self.tags = tags

        self.verify()
