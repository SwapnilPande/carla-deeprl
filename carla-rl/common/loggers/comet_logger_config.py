# Use Base Config from environment to build config
from common.loggers.base_logger_config import BaseLoggerConfig


class BaseCometLoggerConfig(BaseLoggerConfig):
    def __init__(self):
        super().__init__()

        # Comet.ml API key for your account
        self.api_key = None

        # Comet.ml Workspace for the project
        self.workspace = None

        # Comet.ml project name within the workspace
        self.project_name = None

        # Comet.ml Existing experiment key
        # This should only be passed if loading existing experiment
        self.experiment_key = None

        ##### These fields should be populated every experiment
        # Name of the experiment
        # If name already exists in logging directory
        self.experiment_name = None

        # A list of tags to tag the experiment with
        self.tags = None

        # Directory in which to store logs
        self.log_dir = None

        # Whether to clear temporary logs if temp log directory already exists
        self.clear_temp_logs = None

        # Whether to save stdout/stderr to comet
        self.output_logging = None

    def populate(self, experiment_name, tags = [], clear_temp_logs = False):
        self.experiment_name = experiment_name
        self.tags = tags
        self.clear_temp_logs = clear_temp_logs


    def verify(self):
        # None of these need to be passed if we passed experiment key
        if(self.experiment_key is not None):
            super().verify(ignore_keys = ["workspace", "project_name", "tags", "experiment_name", "output_logging"])
        # If none, all other fields must be populated
        else:
            super().verify(ignore_keys = ["experiment_key"])