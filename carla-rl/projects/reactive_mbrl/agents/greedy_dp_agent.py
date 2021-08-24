import numpy as np
import torch
import os
import matplotlib.pyplot as plt

import projects.reactive_mbrl.geometry.transformation as transform
from projects.reactive_mbrl.algorithms.dynamic_programming import (
    ACTIONS,
    num_acts,
    num_steer,
    num_throt,
    OnlineQSolver,
)
from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails
from projects.reactive_mbrl.agents.pid_agent import *
import projects.reactive_mbrl.data.reward_map as reward
import matplotlib.ticker as mticker


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    model = EgoModel()
    model.load_state_dict(model_weights)
    return model


def unpack_waypoint(w):
    return (
        [w.transform.location.x, w.transform.location.y],
        normalize(w.transform.rotation.yaw),
    )


class GreedyDPAgent:
    def __init__(self, npc_predictor, output_path):
        self.npc_predictor = npc_predictor
        self.model = load_ego_model()
        self.output_path = output_path
        self.previous_action = None
        self.index = 0

    def reset(self, scenario_name, waypoints):
        self.index = 0
        self.scenario_path = os.path.join(self.output_path, scenario_name)
        self.previous_action = None
        self.waypoints = waypoints
        os.makedirs(self.scenario_path)

    def predict(self, env, measurements, current_speed, target_speed):
        throt_controller = controller.PIDLongitudinalController()
        self.index += 1
        V, world_points, pixel_xy = reward.calculate_reward_map(env, self.waypoints)
        model = load_ego_model()
        solver = OnlineQSolver(model)
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        base_transform = ego_actor.get_transform()
        ego_yaw = np.radians(base_transform.rotation.yaw)
        speed = 5.0
        V, Q = solver.solve(V, world_points, ego_yaw, speed)

        action_value = Q[int(reward.MAP_SIZE / 2), int(reward.MAP_SIZE/2), :]
        action = ACTIONS[np.argmin(action_value)]

        steer = action[0]

        ego_polygon, predicted_obs = self.predict_obstacles(env)
        d = self.calculate_distance_to_obstacle(env, ego_polygon, predicted_obs)
        desired_target_speed = calculate_desired_target_speed(steer, target_speed, d)
        throt = throt_controller.pid_control(desired_target_speed, current_speed, enable_brake=True)
 
        if d <= 3.0 and current_speed > 0:
            throt = -1.0

        action = np.array([steer, throt])
        # if self.index % 10 == 0:
        #     self.plot_debug_info(
        #         measurements,
        #         V,
        #         action_value,
        #         action,
        #         self.waypoints,
        #         env,
        #     )
        self.previous_action = action

        return action

    def predict_obstacles(self, env):
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        base_transform = ego_actor.get_transform()
        ego_yaw = base_transform.rotation.yaw
        ego_bb = create_bbox(ego_actor.bounding_box.extent)
        ego_bb = rotate_points(ego_bb, ego_yaw)
        ego_bb += extract_loc(ego_actor)
        ego_polygon = Polygon(ego_bb)
        actors = [actor for actor in env.carla_interface.actor_fleet.actor_list
                if "vehicle" in actor.type_id 
                and actor != ego_actor
                and actor.get_transform().location.distance(base_transform.location) < 20
        ]

        if len(actors) <= 0:
            return ego_polygon, []

        vehicles = self.npc_predictor.predict(actors, ego_actor)

        bounding_boxes = np.array(
            [create_bbox(actor.bounding_box.extent) for actor in actors]
        )
        current_vehicles = np.array([extract_loc(actor) for actor in actors])
        
        for i in range(len(actors)):
            yaw = actors[i].get_transform().rotation.yaw
            bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)
        
        current_vehicles = bounding_boxes + current_vehicles[:, None, :]
        vehicles.extend(current_vehicles)
        return ego_polygon, vehicles


    def calculate_distance_to_obstacle(self, env, ego_polygon, vehicles):
        if len(vehicles) <= 0:
            return 100
        
        polygons = [Polygon(vehicle) for vehicle in vehicles]
        
        waypoints = env.carla_interface.next_waypoints[:20]
        
        for waypoint in waypoints:
            location = Point([waypoint.transform.location.x, waypoint.transform.location.y])
            for poly in polygons:
                if location.within(poly):
                    return ego_polygon.distance(poly)
        
        return 100


    def plot_debug_info(
        self,
        info,
        V,
        action_value,
        action,
        route,
        env,
    ):

        topdown = info["sensor.camera.rgb/top"]
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        fig, axes = plt.subplots(1, 5, figsize=(40, 10))
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        base_transform = ego_actor.get_transform()
        ego_yaw = np.radians(base_transform.rotation.yaw)

        action_value = action_value[:num_steer * num_throt]
        action_value = action_value.reshape(num_steer, num_throt)

        ax = axes[0]
        ego_speed = info["speed"]
        ax.annotate(f"ego_speed = {ego_speed}", xy=(0, 0), fontsize=20)
        ax.annotate(f"ego_yaw = {ego_yaw}", xy=(0, 0.25), fontsize=20)
        ax.annotate(f"ref_speed = 5", xy=(0, 0.5), fontsize=20)
        ax.annotate(f"action {action}", xy=(0, 0.7), fontsize=20)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax = axes[1]
        ax.imshow(topdown)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        ax = axes[2]
        im = ax.pcolormesh(V)
        fig.colorbar(im, ax=ax)
        ax.set_title("V")
        
        ax = axes[3]
        im = ax.pcolormesh(action_value)
        fig.colorbar(im, ax=ax)
        ax.set_title("Q")

        ax = axes[4]
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        ego_loc = ego_actor.get_transform().location
        plot_actor(ax, ego_actor, color="red")
        ax.plot(route[:, 0], route[:, 1], "o", color="blue")
        ax.set_xlim(ego_loc.x - 20, ego_loc.x + 20)
        ax.set_ylim(ego_loc.y - 20, ego_loc.y + 20)
    
        plt.savefig(os.path.join(self.scenario_path, f"reward_{self.index}.png"))
        plt.clf()
        plt.close()


def normalizeAngle(angle):
    """
    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def normalize(angle):
    return -(((np.radians(angle) + np.pi) % (2 * np.pi)) - np.pi)


def plot_actor(ax, actor, color="black"):
    bounding_box = get_local_points(actor.bounding_box.extent)
    actor_global = transform.transform_points(actor.get_transform(), bounding_box)
    ax.plot(actor_global[:, 0], actor_global[:, 1], color=color)


def get_local_points(extent):
    return np.array(
        [
            [-extent.x, extent.y, 0, 1],
            [extent.x, extent.y, 0, 1],
            [extent.x, -extent.y, 0, 1],
            [-extent.x, -extent.y, 0, 1],
        ]
    )


def get_closest_waypoint(loc, route):
    min_dist = 10000
    min_wpt = None
    for (wpt_x, wpt_y, wpt_yaw) in route:
        d = np.linalg.norm(loc - np.array([wpt_x, wpt_y]))
        if d < min_dist:
            min_dist = d
            min_wpt = (np.array([wpt_x, wpt_y]), wpt_yaw)
    return min_wpt
