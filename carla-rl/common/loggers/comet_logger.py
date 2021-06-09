from comet_ml import Experiment
# For Existing Experiment
from comet_ml.api import API, APIExperiment

from common.loggers.base_logger import BaseLogger

import os


class CometLogger(BaseLogger):
    def __init__(self, config):
        self.config = config
        self.config.verify()
        # super().__init__(config)
        # Not loading an already existing experiment
        if config.experiment_key is None:
            # Create log directory
            super().__init__(config)
            # Configure comet experiment
            self.logger = Experiment(api_key = self.config.api_key,
                                    workspace = self.config.workspace,
                                    project_name = self.config.project_name,
                                    auto_metric_logging = False)

            # Set name of experiment
            self.logger.set_name(self.experiment_name)

            # Apply all tags
            for tag in self.config.tags:
                self.logger.add_tag(tag)

            # Get the experiment key and save to the file
            self.experiment_key = self.logger.get_key()
            self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")
            with open(self.experiment_key_path, "w") as f:
                f.write(self.experiment_key)

        # Else, load existing experiment using key
        else:
            self.logger = APIExperiment(api_key = self.config.api_key,
                                        previous_experiment = self.config.experiment_key)

            ##### Set up log location
            # First, check if we are running the experiment on the same machine
            current_hostname = os.uname()[1]
            experiment_hostname = self.logger.get_hostname()

            # Variable storing whether we found the experiment locally
            self.experiment_exists_locally = False
            # We are on the same machine as the original experiment
            if(current_hostname == experiment_hostname):
                # Now, see if we can find the experiment log locally already
                # We check the experiment_key file in the log directory to see if it matches

                # Get experiment name from existing comet experiment
                self.experiment_name = self.logger.get_name()

                # Get log_dir from config and cat the experiment name
                self.log_dir = os.path.join(self.config.log_dir, self.experiment_name)

                # Construct file_path for experiment_key
                self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")

                # First, check if log exists and experiment_key file exists
                if(os.path.isdir(self.log_dir) and os.path.isfile(self.experiment_key_path)):
                    with open(self.experiment_key_path,'r') as f:
                        local_exp_key = f.read().strip()

                    # Compare two keys and check if they match
                    if(local_exp_key == self.config.experiment_key):
                        self.experiment_exists_locally = True

            if(self.experiment_exists_locally):
                print("LOGGER: Found experiment logs locally at {}.".format(self.log_dir))

            else:
                print("LOGGER: Could not find experiment in log_dir. Creating new log directory.")

                # Construct new log directory
                self.log_dir = os.path.join(self.config.log_dir, "comet_temp", self.experiment_name)
                # Delete old directory if exists
                if(os.path.isdir(self.log_dir)):
                    print("LOGGER: Temporary log directory exists... Cleaning up old directory")
                    os.rmdir(self.log_dir)

                # Finally, create the directory
                os.makedirs(self.log_dir)
                print("LOGGER: Experiment name - {}".format(self.experiment_name))
                print("LOGGER: Created log directory at {}".format(self.log_dir))

    def log_scalar(self, name, value, step = None):
        """Logs a single scalar value.

        If step is empty, the timestep is not logged with the scalar
        """

        self.logger.log_metric(name, value, step = step)

    def log_hyperparameters(self, hyperparameters):
        """ Logs code hyperparameters.

        hyperparameters is dictionary containing the name (key) and value (value) of each hyperaparameter
        """

        self.logger.log_parameters(hyperparameters)
