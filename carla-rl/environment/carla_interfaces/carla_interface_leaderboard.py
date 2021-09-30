from environment.carla_interfaces.server import CarlaServer
from environment.carla_interfaces import planner
from environment import env_util as util
from abc import ABC
import time
import random
import numpy as np
import threading
import sys, os
from copy import deepcopy
import json

# Leaerboard Import
from leaderboard.leaderboard_evaluator import LeaderboardEvaluator
from leaderboard.utils.statistics_manager import StatisticsManager
import carla



class LeaderboardArgs:
    def __init__(self, agent, agent_config, server_port):
        self.agent = agent
        self.agent_config = agent_config

        self.scenario_class = "train_scenario"
        self.scenarios = os.path.join(os.environ["CARLA_RL"], "assets/all_towns_traffic_scenarios.json")
        self.town = "Town01"
        self.routes = os.path.join(os.environ["CARLA_RL"], "assets/routes_all.xml")
        self.repetitions = 1

        self.host = "localhost"
        self.port = server_port
        self.trafficManagerPort = random.randint(10000, 60000)
        self.trafficManagerSeed = 0 # Deterministic

        self.resume = False

        self.checkpoint = "test.txt"
        self.record = ''

        self.timeout = 60
        self.debug = False



class LeaderboardInterface():

    def __init__(self, config, log_dir):
        self.config = config

        # These are events to sync communication between the leaderboard thread and the main thread
        # When main thread (policy) is sending data to leaderboard, we set the send_event flag
        # When waiting for data from the leaderboard thread, we wait on the receive event
        self.send_event = threading.Event()
        self.receive_event = threading.Event()
        self.data_buffer = {
            "lock" : threading.Lock(),
            "initial_reset" : True
        }

        # This is our leaderboard agent
        self.agent_path = os.path.join(os.environ["CARLA_RL"], "carla-rl/environment/carla_interfaces/rl_agent")
        self.agent_config = {
            "send_event" : self.send_event,
            "receive_event" : self.receive_event,
            "data_buffer" : self.data_buffer,
            "proximity_threshold" : self.config.obs_config.vehicle_proximity_threshold,
            "target_speed" : self.config.action_config.target_speed,
            "action_config" : self.config.action_config,
            "obs_config" : self.config.obs_config
        }


        # Instantiate and start server
        self.server = CarlaServer(config)

        self.setup()

    def setup(self):
        # Start the carla server and get a client
        self.server.start()
        time.sleep(5)


        self.runner = LeaderboardEvaluator(LeaderboardArgs(self.agent_path, self.agent_config, self.server.server_port), StatisticsManager())


        # Start the ScenarioRunner
        # We use the scenario runner to handle actually running the scenarios
        # Note that the scenario runner is started in a thread
        # We do this to handle the interface between a gym environment (which expects a callback to step the environment)
        # and the leaderboard evaluator (which expects a callback to step the policy)
        self.leaderboard_thread = threading.Thread(target = lambda: self.runner.run(LeaderboardArgs(self.agent_path, self.agent_config, self.server.server_port)))
        self.leaderboard_thread.start()


    def reset(self, unseen = False, index = 0):
        # Tell the agent to reset and start the next scenario
        self.data_buffer["lock"].acquire()
        self.data_buffer["reset"] = True
        self.data_buffer["lock"].release()

        self.send_event.set()


        # Wait for the leaderboard thread to send data about the current obs
        self.receive_event.wait()
        self.receive_event.clear()

        self.data_buffer["lock"].acquire()

        # obs = deepcopy(self.data_buffer["leaderboard_data"])
        for key in self.data_buffer["leaderboard_data"]:
            if(isinstance(self.data_buffer["leaderboard_data"][key], np.ndarray)):
                self.data_buffer["leaderboard_data"][key] = self.data_buffer["leaderboard_data"][key].tolist()
        # obs = deepcopy(json.loads(json.dumps(self.data_buffer["leaderboard_data"]))
        obs = deepcopy(self.data_buffer["leaderboard_data"])
        self.data_buffer["lock"].release()

        return obs


    def step(self, action):
        # print("STARTED STEP")

        self.data_buffer["lock"].acquire()
        self.data_buffer["policy_action"] = action
        self.data_buffer["lock"].release()

        # print("SENT DATA TO THREAD")
        self.send_event.set()

        # Wait for the leaderboard thread to send data about the current obs
        self.receive_event.wait()
        self.receive_event.clear()

        # print("RECEIVED DATA FROM THREAD")
        self.data_buffer["lock"].acquire()

        # obs = deepcopy(self.data_buffer["leaderboard_data"])

        for key in self.data_buffer["leaderboard_data"]:
            if(isinstance(self.data_buffer["leaderboard_data"][key], np.ndarray)):
                self.data_buffer["leaderboard_data"][key] = self.data_buffer["leaderboard_data"][key].tolist()

        # obs = json.loads(json.dumps(self.data_buffer["leaderboard_data"]))
        obs = deepcopy(self.data_buffer["leaderboard_data"])
        self.data_buffer["lock"].release()

        return obs


    # def get_actor_list(self):
    #     return []#self.data_buffer["world"].get_actors()

    # def get_ego_vehicle(self):
    #     return self.data_buffer["ego_vehicle"]

    # def get_traffic_actors(self):
    #     return []#self.data_buffer["world"].get_actors().filter('*traffic_light*')

    # def get_map(self):
    #     return self.data_buffer["map"]


    def close(self):
        if self.server is not None:
            self.server.close()