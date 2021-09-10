import os
import sys
import glob
import traceback


class BaseLogger:
    def __init__(self, config):
        self.logger = None
        self.config = config

        self.create_log_file()

        if(self.config.pdb_on_exception):
            self.pdb_on_exception()

    def create_log_file(self):
        # Create the file to log all data in
        self.experiment_name = self.config.experiment_name
        self.log_dir = os.path.join(self.config.log_dir, self.experiment_name)

        if(os.path.isdir(self.log_dir)):
            # Find all files with the same experiment name
            matching_files = glob.glob(self.log_dir + "_*")
            # Append a number to the experiment name
            self.experiment_name = self.config.experiment_name + "_{}".format(len(matching_files))
            self.log_dir = os.path.join(self.config.log_dir, self.experiment_name)

        # Create log dir
        os.makedirs(self.log_dir)

        print("LOGGER: Experiment name - {}".format(self.experiment_name))
        print("LOGGER: Created log directory at {}".format(self.log_dir))

    def pdb_on_exception(self, debugger = "ipdb", limit = 100):
        """Install handler attach post-mortem pdb console on an exception."""

        def pdb_excepthook(exc_type, exc_val, exc_tb):
            traceback.print_tb(exc_tb, limit=limit)
            __import__(str(debugger).strip().lower()).post_mortem(exc_tb)

        sys.excepthook = pdb_excepthook




    def log_scalar(self, name, value, step):
        raise NotImplementedError