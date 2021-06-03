import os
import signal
import subprocess
import traceback
import random
import time

from tqdm import tqdm

class CarlaServer():
    def __init__(self, config=None):
        # print("Launching CARLA server...")
        # Save config parameters
        self.config = config
        self.server_port = config.server_port
        self.server_binary = config.server_binary
        self.carla_gpu = config.carla_gpu
        self.render_server = config.render_server
        self.live_carla_processes = set()

        self.server_process = None

        if self.server_port == -1:
            self.server_port = random.randint(10000, 60000)
        else:
            pass

        # Configure environment variables
        self.carla_env = os.environ.copy()
        if self.carla_gpu is not None:
            self.carla_env["SDL_HINT_CUDA_DEVICE"] = str(self.carla_gpu)
            # Delete cuda visible devices
            try:
                del self.carla_env['CUDA_VISIBLE_DEVICES']
            except:
                pass

        if not self.render_server:
            os.environ["SDL_VIDEODRIVER"] = "offscreen"

        # Command to launch carla
        self.launch_command = [
                self.server_binary, "-carla-rpc-port={}".format(self.server_port)
        ]


    def _attempt_server_launch(self):

        # import ipdb; ipdb.set_trace()
        # Try to launch
        print("Attempting to start carla on GPU {0}".format(self.carla_gpu))
        print(type(self.launch_command[0]))
        print(type(self.launch_command[1]))
        print(self.launch_command)
        try:
            self.server_process = subprocess.Popen(self.launch_command,
                preexec_fn=os.setsid, env=self.carla_env)
        except Exception as e:
            print("Error in starting carla server : {}".format(traceback.format_exc()))

            # Clean up if necessary
            self.close()

        # Return true if success
        if self.server_process:
            print("Launched server at port:", self.server_port)

            server_launch_delay = 25
            print('Waiting {}s for server to finish setting up'.format(server_launch_delay))
            for _ in tqdm(range(server_launch_delay)):
                time.sleep(1)

            return True




        return False


    def start(self):
        """Starts the CARLA server.

        Will attempt to start the CARLA server for number of retries specified in config.
        """
        server_started = False
        server_start_retries = 0

        while ((not server_started) and server_start_retries < self.config.server_retries):
            server_started = self._attempt_server_launch()
            server_start_retries += 1

        # Max retry attempts exceeded
        if(not server_started):
            raise Exception("Failed to start CARLA server. Check configuration and try again.")

        # time.sleep(12)

    def __del__(self):
        self.close()

    def close(self):
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            self.server_port = None
            self.server_process = None
            print("Killed server process")
