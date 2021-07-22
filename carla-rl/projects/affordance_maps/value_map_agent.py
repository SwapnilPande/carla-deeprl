import traceback
import math

from shapely.geometry import Point, Polygon

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import cv2
import matplotlib.pyplot as plt
import carla
import torch

from client_bounding_boxes import ClientSideBoundingBoxes
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.action_configs import *
from environment.config.scenario_configs import *
from train_value import RecurrentValueNetwork
from train_ego import EgoModel
from value_map import ACTIONS, YAWS, SPEEDS
from common.utils import preprocess_rgb


VALUE_NETWORK_PATH = '/home/brian/temp/recurrent/2021-06-29_04-25-53/checkpoints/epoch=29-step=177659.ckpt'
EGO_MODEL_PATH = '/home/brian/WorldOnRails/ego_model.th'
CALIBRATION = np.array([[8, 0, 8], [0, 8, 8], [0,  0,  1]])


class ValueMapAgent:
    def __init__(self, env):
        self.value_model = RecurrentValueNetwork.load_from_checkpoint(VALUE_NETWORK_PATH)
        self.value_map = None
        self.value_hidden_state = None

        self.ego_model = EgoModel(dt=1/4.)
        ego_weights = torch.load(EGO_MODEL_PATH)
        self.ego_model.load_state_dict(ego_weights)

        # don't technically need env, but makes life easier for now
        self.env = env
        self.ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
        self.map_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/map'].sensor

    def get_world_points(self):
        """
        Fetch world coordinates corresponding to 16x16 value map
        """
        base_transform = self.ego_actor.get_transform()
        base_waypoint = self.env.carla_interface.map.get_waypoint(base_transform.location, project_to_road=True)

        pixel_x, pixel_y = np.meshgrid(np.arange(16), np.arange(16))
        pixel_xy = np.stack([pixel_x.flatten(), pixel_y.flatten(), np.ones(16*16)], axis=-1)
        world_pts = np.linalg.inv(CALIBRATION).dot(pixel_xy.T).T[:,:2]

        yaw = -(((np.radians(base_transform.rotation.yaw) + np.pi) % (2*np.pi)) - np.pi)
        rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        world_pts = world_pts.dot(rot_matrix)

        world_pts *= self.map_actor.get_transform().location.z
        world_pts[:,0] += self.map_actor.get_transform().location.x
        world_pts[:,1] += self.map_actor.get_transform().location.y

        return world_pts

    def update(self, image):
        """
        Use new observation to update value map
        """
        image = preprocess_rgb(image)[None,None]
        value_map, self.value_hidden_state = self.value_model(image, self.value_hidden_state)
        value_map = value_map.squeeze().permute(1,2,0)
        self.value_map = value_map.reshape(16,16,4,5)

    def get_action(self):
        """
        Plan optimal action under current value map
        """
        tf = self.ego_actor.get_transform()
        loc = [tf.location.x, tf.location.y]
        yaw = tf.rotation.yaw
        vel = self.ego_actor.get_velocity()
        spd = np.linalg.norm([vel.x, vel.y])

        Qs = self.get_Q_values(loc, yaw, spd, ACTIONS)
        action = ACTIONS[Qs.argmax()]
        # print(action)
        return action

    def get_Q_values(self, loc, yaw, spd, actions):
        """
        Use value map and ego model to compute Q values
        """
        loc = torch.FloatTensor(loc)
        yaw = torch.FloatTensor([yaw])
        spd = torch.FloatTensor([spd])
        actions = torch.FloatTensor(actions)

        next_locs, next_yaws, next_spds = self.ego_model.forward(
                    loc[None,:].repeat(28,1).reshape(-1,2),
                    yaw[None].repeat(28,1).reshape(-1,1),
                    spd[None].repeat(28,1).reshape(-1,1),
                    actions.repeat(1,1).reshape(-1,2))

        locs = self.get_world_points()
        values = self.value_map.detach().numpy()

        # normalize grid so we can use grid interpolation
        offset = locs[0]
        _locs = locs - offset
        theta = np.arctan2(_locs[-1][1], _locs[-1][0])
        _locs = rotate_pts(_locs, (np.pi/4)-theta)

        # set up grid interpolator
        min_x, min_y = np.min(_locs, axis=0)
        max_x, max_y = np.max(_locs, axis=0)
        xs, ys = np.linspace(min_x, max_x, 16), np.linspace(min_y, max_y, 16)
        values = np.moveaxis(values, 0, 1) # because indexing=ij, for more: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
        grid_interpolator = RegularGridInterpolator((xs, ys, SPEEDS, YAWS), values, bounds_error=False, fill_value=None)

        # interpolate Q values
        next_locs = next_locs.detach().numpy()
        next_spds = next_spds.detach().numpy()
        next_yaws = next_yaws.detach().numpy()

        _next_locs = next_locs - offset
        _next_locs = rotate_pts(_next_locs, (np.pi/4)-theta)
        pred_pts = np.concatenate([_next_locs, next_spds, next_yaws], axis=1)
        Qs = grid_interpolator(pred_pts, method='linear')

        # add speed rewards
        spd_rewards = (next_spds * 1e-2).flatten()
        waypoint_rewards = self.get_waypoint_rewards(next_locs)
        # Qs = Qs + spd_rewards.flatten() + waypoint_rewards
        Qs = waypoint_rewards + spd_rewards.flatten()
        # print(Qs)
        return Qs

    def get_waypoint_rewards(self, locs):
        # get next two waypoints
        waypoint_queue = self.env.carla_interface.global_planner._waypoints_queue
        waypoints = []
        for wp, _, _ in waypoint_queue:
            waypoints.append(wp)
            if len(waypoints) == 2:
                break

        dists = []
        for loc in locs:
            loc = loc.astype(np.float64)
            dist = self.env.carla_interface.global_planner.getPointToLineDistance(
                carla.Transform(location=carla.Location(x=loc[0], y=loc[1])),
                waypoints[0],
                waypoints[1]
            )
            dists.append(dist)

        return -np.array(dists)


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


def main():
    config = DefaultMainConfig()
    config.server_fps = 20
    obs_config = PerspectiveRGBObservationConfig() # LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'64',
        'sensor_y_res':'64',
        'fov':'90', \
        'sensor_tick': '0.0'}
    obs_config.sensors['sensor.camera.rgb/map'] = {
        'x':13.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'16',
        'sensor_y_res':'16',
        'fov':'90', \
        'sensor_tick': '0.0'}

    scenario_config = NoCrashDenseTown01Config()

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(observation_config=obs_config)
    env = CarlaEnv(config=config)

    obs = env.reset()
    # for _ in range(25):
    #     env.step(np.array([0,-1]))

    image = env.render(camera='sensor.camera.rgb/top')

    agent = ValueMapAgent(env)

    plt.ion()
    fig, axs = plt.subplots(4,5)

    for _ in range(10000):
        agent.update(image)
        action = agent.get_action()
        obs, reward, done, info = env.step(action)
        image = env.render(camera='sensor.camera.rgb/top')

        values = agent.value_map.detach().numpy()
        for s in range(4):
            for y in range(5):
                axs[s,y].imshow(values[:,:,s,y])
        plt.show()
        plt.pause(.001)


        print(reward, done)

        if done:
            break





if __name__ == '__main__':
    main()