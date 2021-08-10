import datetime
import os
import numpy as np
import cv2
import json
import torch

from projects.npc_modeling.data.actor_manager import ActorManager


class DataCollector:
    def __init__(self, env, path):
        self.env = env
        self.path = path
        self.waypointer = None

    def collect_trajectory(self, speed, max_path_length=5000, warm_start=0):
        print(f"Collecting a new trajectory.")
        self.setup_output_dir()
        obs = self.env.reset()
        route = self.get_route()
        actor_manager = ActorManager()

        for step in range(max_path_length):
            expert_action = self.env.get_autopilot_action(speed)
            next_obs, reward, done, info = self.env.step(expert_action)

            experience = create_experience(
                obs, next_obs, expert_action, done, reward, info
            )

            actor_manager.step(
                self.env.carla_interface.actor_fleet.actor_list,
                self.env.carla_interface.actor_fleet.sensor_manager.sensors[
                    "sensor.camera.rgb/top"
                ],
                info,
            )
            self.save_data(experience, info, step)
            if done:
                break

            obs = next_obs

        dead_actors = [actor_manager.actors[id] for id in actor_manager.actors]
        for actor in dead_actors:
            self.save_actor_data(actor)
        print(f"Done collecting trajectory, number of steps {step + 1}")
        return step + 1

    def get_route(self):
        waypoints = self.env.carla_interface.global_planner._waypoints_queue
        return np.array(
            [[w[0].transform.location.x, w[0].transform.location.y] for w in waypoints]
        )

    def setup_output_dir(self):
        now = datetime.datetime.now()
        salt = np.random.randint(100)
        fname = "_".join(
            map(
                lambda x: "%02d" % x,
                (now.month, now.day, now.hour, now.minute, now.second, salt),
            )
        )
        self.save_path = os.path.join(self.path, fname)
        print(f"Setting up output directory {self.save_path}")

        # check for conflicts
        if os.path.isdir(self.save_path):
            print("Directory conflict, trying again...")
            return 0

        os.mkdir(self.save_path)
        os.mkdir(os.path.join(self.save_path, "topdown"))
        os.mkdir(os.path.join(self.save_path, "topdown_seg"))
        os.mkdir(os.path.join(self.save_path, "measurements"))

    def save_data(self, experience, info, idx):
        # self.save_img(info["sensor.camera.rgb/top"], "topdown", idx)
        # self.save_img(
        # info["sensor.camera.semantic_segmentation/top"], "topdown_seg", idx
        # )
        measurements_path = os.path.join(
            self.save_path, "measurements", "{:04d}.json".format(idx)
        )
        with open(measurements_path, "w") as out:
            json.dump(experience, out)

    def save_img(self, img_arr, name, idx):
        img_path = os.path.join(self.save_path, name, "{:04d}.png".format(idx))
        cv2.imwrite(img_path, img_arr)

    def save_actor_data(self, dead_track):
        actor_path = os.path.join(self.save_path, str(dead_track.id))
        assert not os.path.exists(actor_path)
        os.mkdir(actor_path)
        self.save_np(actor_path, dead_track.transforms, "transforms.npy")
        self.save_np(actor_path, dead_track.velocities, "velocities.npy")
        self.save_np(actor_path, dead_track.accelerations, "accelerations.npy")
        self.save_np(actor_path, dead_track.controls, "controls.npy")

    def save_np(self, path, value, name):
        with open(os.path.join(path, name), "wb") as f:
            np.save(f, value)


def create_experience(obs, next_obs, action, done, reward, info):

    return {
        "obs": obs.tolist(),
        "next_obs": next_obs.tolist(),
        "action": action.tolist(),
        "reward": reward,
        "done": done.item(),
        "location": info["location"],
        "speed": info["speed"],
        "yaw": info["ego_vehicle_theta"],
        "distance_to_goal": info["distance_to_goal"],
        "speed_reward": info["speed_reward"],
        "steer_reward": info["steer_reward"],
        "ego_vehicle_x": info["ego_vehicle_x"],
        "ego_vehicle_y": info["ego_vehicle_y"],
        #'gnss': info['sensor.other.gnss']
    }
