from torch.utils.tensorboard import SummaryWriter
from common.loggers.base_logger import BaseLogger

from glob import glob
import os
import shutil
import torch
import pickle
from uuid import uuid4

class TensorboardLogger(BaseLogger):
    def __init__(self, config):
        # Verify config before using it
        self.config = config
        self.config.verify()

        # Not loading an already existing experiment
        if config.experiment_key is None:
            # Create log directory
            super().__init__(config)

            # Generate unique experiment key save to the file
            self.experiment_key = str(uuid4())
            self.experiment_key_path = os.path.join(self.log_dir, "experiment_key")
            with open(self.experiment_key_path, "w") as f:
                f.write(self.experiment_key)

            self.experiment_exists_locally = True

        # Else, load existing experiment using key
        else:
            ##### Set up log directory
            # Variable storing whether we found the experiment locally
            self.experiment_exists_locally = False

            # Check if experiment_key file exists
            for dir in glob(os.path.join(self.config.log_dir, "*", "")):
                experiment_key_path = os.path.join(dir, "experiment_key")
                if(os.path.isfile(experiment_key_path)):
                    with open(experiment_key_path,'r') as f:
                        local_exp_key = f.read().strip()

                    # Compare two keys and check if they match
                    if(local_exp_key == self.config.experiment_key):
                        self.experiment_exists_locally = True
                        break
            if(self.experiment_exists_locally):
                self.log_dir = dir
                _, self.experiment_name = os.path.split(os.path.normpath(self.log_dir))

                print("LOGGER: Found experiment logs locally at {}.".format(self.log_dir))
            else:
                raise Exception("Could not find experiment in log directory")

        self.logger = SummaryWriter(log_dir = os.path.join(self.log_dir, "tensorboard"))

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



    def is_log_local(self):
        return self.experiment_exists_locally

    def log_scalar(self, name, value, step = None):
        """Logs a single scalar value.

        If step is empty, the timestep is not logged with the scalar
        """

        self.logger.add_scalar(name, value, step)

    def log_hyperparameters(self, hyperparameters):
        """ Logs code hyperparameters.

        hyperparameters is dictionary containing the name (key) and value (value) of each hyperaparameter
        """
        assert isinstance(hyperparameters, dict)

        # Check if hyperparameters.pkl already exists
        if(os.path.isfile(os.path.join(self.log_dir, "hyperparameters.pkl"))):
            logged_hyperparameters = pickle.load(open(os.path.join(self.log_dir, "hyperparameters.pkl"), "rb"))

            # Combine logged hyperparameters with new ones
            hyperparameters = {**logged_hyperparameters, **hyperparameters}

        file_path = os.path.join(self.log_dir, "hyperparameters.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(hyperparameters, f)


    def prep_dir(self, dir):
        # Get full log path within the log directory
        full_log_path = os.path.join(self.log_dir, dir)

        # If this subdirectory does not exist, create it first
        if(not os.path.isdir(full_log_path)):
            os.makedirs(full_log_path)

        return full_log_path

    def torch_save(self, state_dict, log_path, model_name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        # Save the file using torch.save
        file_path = os.path.join(full_log_path, model_name)
        torch.save(state_dict, file_path)


    def torch_load(self, log_path, name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        with open(file_path, 'rb') as f:
            obj = torch.load(f, map_location='cpu')

        return obj

    def pickle_save(self, obj, log_path, name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, -1)


    def pickle_load(self, log_path, name):

        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        with open(file_path, 'rb') as f:
            obj = pickle.load(f)

        return obj

    def other_load(self, log_path, name):
        # Get full path to log directory
        full_log_path = self.prep_dir(log_path)

        file_path = os.path.join(full_log_path, name)

        return file_path




