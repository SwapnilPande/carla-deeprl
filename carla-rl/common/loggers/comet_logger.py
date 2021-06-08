from comet_ml import Experiment
from common.loggers.base_logger import BaseLogger


class CometLogger(BaseLogger):
    def __init__(self, config):
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
