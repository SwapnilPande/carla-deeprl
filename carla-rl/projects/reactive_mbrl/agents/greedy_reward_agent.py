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
)
from projects.reactive_mbrl.ego_model import EgoModel, EgoModelRails
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


class GreedyRewardAgent:
    def __init__(self, output_path):
        self.model = load_ego_model()
        self.output_path = output_path
        self.previous_action = None
        self.index = 0

    def reset(self, scenario_name):
        self.index = 0
        self.scenario_path = os.path.join(self.output_path, scenario_name)
        self.previous_action = None
        os.makedirs(self.scenario_path)

    def predict(self, measurements, env):
        self.index += 1
        route = env.carla_interface.next_waypoints

        locs = np.array([measurements["ego_vehicle_x"], measurements["ego_vehicle_y"]])
        if len(route) == 1:
            closest_waypoint = unpack_waypoint(route[0])
        elif len(route) >= 2:
            closest_waypoint = unpack_waypoint(route[1])
        else:
            return self.previous_action
        locs = np.tile(locs, (num_acts, 1))

        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        base_transform = ego_actor.get_transform()
        ego_yaw = np.radians(base_transform.rotation.yaw)
        yaws = np.array([ego_yaw])
        yaws = np.tile(yaws, (num_acts, 1))
        speeds = np.array([measurements["speed"]])
        # speeds = np.array([5])
        speeds = np.tile(speeds, (num_acts, 1))

        locs = torch.tensor(locs)
        yaws = torch.tensor(yaws)
        speeds = torch.tensor(speeds)
        actions = torch.tensor(ACTIONS)

        pred_locs, pred_yaws, pred_speeds = self.model.forward(
            locs, yaws, speeds, actions
        )
        pred_locs = pred_locs.detach().numpy()
        pred_yaws = pred_yaws.detach().numpy()
        pred_speeds = pred_speeds.detach().numpy()
        (
            loc_loss,
            yaw_loss,
            speed_loss,
            lane_loss,
            collision_loss,
            action_value,
        ) = reward.calculate_action_value_map(
            pred_locs, pred_yaws, pred_speeds, closest_waypoint, env
        )
        action_value = action_value[:num_acts]
        # action_value = torch.tensor(action_value)
        # _, action_indices = action_value.topk(3, largest=False)
        # action = ACTIONS[action_indices].mean(axis=0)
        action = ACTIONS[np.argmin(action_value)]

        # action_value = action_value.detach().numpy()
        # steer, throt = action
        # action = np.array([-steer, throt])

        if self.index % 20 == 0:
            self.plot_debug_info(
                loc_loss,
                yaw_loss,
                speed_loss,
                lane_loss,
                collision_loss,
                measurements,
                pred_locs,
                action_value,
                route,
                closest_waypoint,
                yaws[0].detach().numpy(),
                action,
                env,
            )
        self.previous_action = action

        return action

    def plot_debug_info(
        self,
        loc_loss,
        yaw_loss,
        speed_loss,
        lane_loss,
        collision_loss,
        info,
        pred_locs,
        action_value,
        route,
        closest_waypoint,
        ego_yaw,
        action,
        env,
    ):

        route = np.array(
            [
                [
                    w.transform.location.x,
                    w.transform.location.y,
                    normalize(w.transform.rotation.yaw),
                ]
                for w in route
            ]
        )

        action_value = action_value[:num_acts].reshape(num_steer, num_throt)
        loc_loss = loc_loss[:num_acts].reshape(num_steer, num_throt)
        yaw_loss = yaw_loss[:num_acts].reshape(num_steer, num_throt)
        speed_loss = speed_loss[:num_acts].reshape(num_steer, num_throt)
        lane_loss = lane_loss[:num_acts].reshape(num_steer, num_throt)
        collision_loss = collision_loss[:num_acts].reshape(num_steer, num_throt)

        topdown = info["sensor.camera.rgb/top"]
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        fig, axes = plt.subplots(3, 4, figsize=(35, 30))

        ax = axes[0, 0]
        ego_speed = info["speed"]
        ax.annotate(f"ego_speed = {ego_speed}", xy=(0, 0), fontsize=20)
        ax.annotate(f"ego_yaw = {ego_yaw}", xy=(0, 0.25), fontsize=20)
        ax.annotate(f"ref_speed = 5", xy=(0, 0.5), fontsize=20)
        ax.annotate(f"action {action}", xy=(0, 0.7), fontsize=20)
        ax.annotate(f"ref_yaw = {closest_waypoint[1]}", xy=(0, 0.9), fontsize=20)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax = axes[0, 1]
        ax.imshow(topdown)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ticks_loc = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
        ax = axes[0, 2]
        im = ax.pcolormesh(action_value)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Total loss")

        ax = axes[0, 3]
        im = ax.pcolormesh(collision_loss)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Collision loss")

        ax = axes[1, 0]
        im = ax.pcolormesh(loc_loss)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Loc loss")

        ax = axes[1, 1]
        im = ax.pcolormesh(yaw_loss)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Yaw loss")

        ax = axes[1, 2]
        im = ax.pcolormesh(speed_loss)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Speed loss")

        ax = axes[1, 3]
        im = ax.pcolormesh(lane_loss)
        fig.colorbar(im, ax=ax)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        ax.set_yticklabels(ticks_loc)
        ax.set_title("Lane loss")

        ax = axes[2, 0]
        ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        ego_loc = ego_actor.get_transform().location
        plot_actor(ax, ego_actor, color="red")
        boxes = reward.get_actor_polygons(env)
        for box in boxes:
            ax.plot(box[:, 0], box[:, 1], color="blue")
        # for actor in env.carla_interface.actor_fleet.actor_list:
        #     if "vehicle" in actor.type_id and actor != ego_actor:
        #         plot_actor(ax, actor, color="blue")
        ax.plot(route[:, 0], route[:, 1], "o", color="blue")
        ax.plot(closest_waypoint[0][0], closest_waypoint[0][1], "o", color="green")
        ax.plot(pred_locs[:, 0], pred_locs[:, 1], "o", color="red")
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

