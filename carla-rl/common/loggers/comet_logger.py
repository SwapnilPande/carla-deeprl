from comet_ml import Experiment, experiment
# For Existing Experiment
from comet_ml.api import API, APIExperiment

from common.loggers.base_logger import BaseLogger
from common.loggers.logger_utils import lock_open, lock_open_dir

import os
import sys
import shutil
import torch
import pickle
import requests

class CometLogger(BaseLogger):
    def __init__(self, config):
        self.config = config
        self.config.verify()
        # super().__init__(config)
        # Creating a new experiment, not loading existing one
        if config.experiment_key is None:
            # Create log directory
            super().__init__(config)

            # Check if no_logger is passed as command line argument
            # If true, set disabled = True
            disabled = "--no_logger" in sys.argv
            # Configure comet experiment
            self.logger = Experiment(api_key = self.config.api_key,
                                    workspace = self.config.workspace,
                                    project_name = self.config.project_name,
                                    auto_metric_logging = True,
                                    disabled = disabled,
                                    auto_output_logging=self.config.output_logging)

            # Set name of experiment
            self.logger.set_name(self.experiment_name)

            # Apply all tags
            for tag in self.config.tags:
                self.logger.add_tag(tag)

            # Get the experiment key and save to the file
            self.experiment_key = self.logger.get_key()
            self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")
            with lock_open(self.experiment_key_path, "w") as f:
                f.write(self.experiment_key)

            # Flag to determine if this is a temporary log
            # If not a temp log, there is no need to ever download files
            # If it is a temp log, we may need to download files
            self.is_temp_log = False

        # Else, load existing experiment using key
        else:
            # Don't call super constructor, we don't want to create a new directory

            # Create an experiment object
            self.logger = APIExperiment(api_key = self.config.api_key,
                                        previous_experiment = self.config.experiment_key)

            # Get experiment name from existing comet experiment
            self.experiment_name = self.logger.get_name()

            # Get assets that are available online
            self.get_available_assets()

            ##### Set up log location
            # Check if we are running the experiment on the same machine
            current_hostname = os.uname()[1]
            experiment_hostname = self.logger.get_hostname()

            # Variable storing whether we found the experiment locally
            self.is_temp_log = True


            # We are on the same machine as the original experiment
            if(current_hostname == experiment_hostname):
                # Check if we can find the experiment log locally already
                # We check the experiment_key file in the log directory to see if it matches

                # Get log_dir from config and cat the experiment name
                self.log_dir = os.path.join(self.config.log_dir, self.experiment_name)

                # Construct file_path for experiment_key
                self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")

                # First, check if log exists and experiment_key file exists
                if(os.path.isdir(self.log_dir) and os.path.isfile(self.experiment_key_path)):
                    with lock_open(self.experiment_key_path,'r') as f:
                        local_exp_key = f.read().strip()

                    # Compare two keys and check if they match
                    if(local_exp_key == self.config.experiment_key):
                        self.is_temp_log = False

            if(not self.is_temp_log):
                print("LOGGER: Found experiment logs locally at {}.".format(self.log_dir))

            else:
                print("LOGGER: Could not find experiment in log_dir. Checking for experiment in temp directory.")

                # Construct new log directory
                self.log_dir = os.path.join(self.config.log_dir, "comet_temp", self.experiment_name)
                # Delete old directory if exists
                if(os.path.isdir(self.log_dir)):
                    # Confirm that it actually is the same experiment by checking the experiment_key file
                    # Construct file_path for experiment_key
                    self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")

                    # First, check if experiment_key file exists
                    if(os.path.isfile(self.experiment_key_path)):
                        with lock_open(self.experiment_key_path, 'r') as f:
                            local_exp_key = f.read().strip()

                        # If keys match and the user doesn't want to delete the old log
                        if(local_exp_key == self.config.experiment_key and not self.config.clear_temp_logs):
                            print("LOGGER: Temporary log directory exists, not cleaning up because clear_temp_logs flag is not set")

                    # Else, delete old directory, regardless of whether it is the correct experiment.
                    if(self.config.clear_temp_logs):
                        print("LOGGER: Temporary log directory exists... Cleaning up old directory")
                        shutil.rmtree(self.log_dir)
                        # Finally, create the directory
                        os.makedirs(self.log_dir)


                print("LOGGER: Experiment name - {}".format(self.experiment_name))
                print("LOGGER: Created log directory at {}".format(self.log_dir))

    def get_available_assets(self):
        asset_list = self.logger.get_asset_list()

        self.comet_assets = {}

        for asset in asset_list:
            # This is a non-model asset
            if(asset["type"] == None and asset["dir"] == "others"):
                self.comet_assets[asset["fileName"]] = {
                    "id" : asset["assetId"]
                }
            # This is a model aset
            elif(asset["type"] == "model-element"):
                # Remove leading models/ directory from path
                # This is a comet specific directory we don't want in the local directory
                log_path = os.path.join(*asset["dir"].split(os.path.sep)[1:])
                file_path = os.path.join(log_path, asset["fileName"])
                self.comet_assets[file_path] = {
                    "id" : asset["assetId"]
                }

            # Else, ignore all other file types for now



    def is_log_temp(self):
        return self.is_temp_log

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

    def log_model(self, comet_path, file):
        self.logger.log_model(comet_path, file)


    def log_asset(self, comet_path, file):
        _, file_name = os.path.split(file)
        if self.config.experiment_key is None:
            self.logger.log_asset(file, file_name = os.path.join(comet_path, file_name))
        else:
            self.logger.log_asset(filename = file, ftype = comet_path)

    def prep_dir(self, dir):
        # Get full log path within the log directory
        full_log_path = os.path.join(self.log_dir, dir)

        # If this subdirectory does not exist, create it first
        with lock_open_dir(full_log_path) as f:
            if(not os.path.isdir(full_log_path)):
                os.makedirs(full_log_path)

        return full_log_path


    def comet_download(self, asset_path):
        comet_url = "https://www.comet.ml/api/rest/v2/experiment/asset/get-asset?experimentKey={experiment_key}&assetId={asset_id}"
        # Check if asset in comet assets
        if(asset_path in self.comet_assets):
            # Relative path for logging the asset
            asset_log_path, file_name = os.path.split(asset_path)
            # Get the absolute path and make sure dir exists
            full_log_path = self.prep_dir(asset_log_path)

            # Full download path
            file_path = os.path.join(full_log_path, file_name)

            # Add experiment key and asset id to url
            comet_url = comet_url.format(experiment_key = self.config.experiment_key,
                                        asset_id = self.comet_assets[asset_path]['id'])
            # Construct header for auth
            headers = {"Authorization" : self.config.api_key}

            # Create request in streaming mode
            with requests.get(comet_url, stream=True, headers = headers) as r:
                r.raise_for_status()
                with lock_open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        # If you have chunk encoded response uncomment if
                        # and set chunk_size parameter to None.
                        #if chunk:
                        f.write(chunk)
        else:
            raise Exception("Requested asset does not exist on comet or is not supported currently by the logger.")

    def torch_save(self, state_dict, log_path, model_name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        # Save the file using torch.save
        file_path = os.path.join(full_log_path, model_name)
        torch.save(state_dict, file_path)

        # Lastly, log model to comet as well
        self.log_model(log_path, file_path)

    def torch_load(self, log_path, name, map_location = 'cpu'):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        # Dowload file if not local
        if(self.is_temp_log):
            self.comet_download(os.path.join(log_path, name))

        with lock_open(file_path, 'rb') as f:
            obj = torch.load(f, map_location=map_location)

        return obj

    def pickle_save(self, obj, log_path, name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        with lock_open(file_path, 'wb') as f:
            pickle.dump(obj, f, -1)

        # Lastly, log asset to comet as well
        self.log_asset(log_path, file_path)

    def pickle_load(self, log_path, name):

        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        # Dowload file if not local
        if(self.is_temp_log):
            self.comet_download(os.path.join(log_path, name))

        with lock_open(file_path, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def other_load(self, log_path, name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        # Dowload file if not local
        if(self.is_temp_log):
            self.comet_download(os.path.join(log_path, name))

        return file_path




