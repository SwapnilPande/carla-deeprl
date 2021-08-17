import os
import sys
import json
import datetime
import argparse
import random
import math

import numpy as np
import cv2
from omegaconf import OmegaConf
import carla
from shapely.geometry import Point, Polygon

# from carla_env import CarlaEnv
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *
from environment.config.scenario_configs import *
from environment.config.action_configs import *


def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


def collect_trajectory(env, save_dir, speed=.5, max_path_length=5000):
    now = datetime.datetime.now()
    salt = np.random.randint(100)
    fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
    save_path = os.path.join(save_dir, fname)
    rgb_path = os.path.join(save_path, 'rgb')
    segmentation_path = os.path.join(save_path, 'segmentation')
    topdown_path = os.path.join(save_path, 'topdown')
    reward_path = os.path.join(save_path, 'reward')
    world_path = os.path.join(save_path, 'world')
    measurements_path = os.path.join(save_path, 'measurements')

    # check for conflicts
    if os.path.isdir(save_path):
        print('Directory conflict, trying again...')
        return 0
    
    # make directories
    os.mkdir(save_path)
    os.mkdir(rgb_path)
    os.mkdir(segmentation_path)
    os.mkdir(topdown_path)
    os.mkdir(reward_path)
    os.mkdir(world_path)
    os.mkdir(measurements_path)

    obs = env.reset()

    calibration = np.array([[32, 0, 32],
                            [0, 32, 32],
                            [0,  0,  1]])

    # calibration = np.array([[64, 0, 64],
    #                         [0, 64, 64],
    #                         [0,  0,  1]])
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    # camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/map'].sensor
    # camera_actor.calibration = calibration

    for _ in range(25):
        action = env.get_autopilot_action(speed)
        _ = env.step(action)

    for step in range(max_path_length):
        if args.use_autopilot:
            action = env.get_autopilot_action(speed)
        else:
            action = np.random.uniform([-.5,-1.],[.5,1.], (2,))

        if args.use_autopilot and args.apply_noise:
            noise = 1e-2 * np.random.randn()
            action[0] += noise
            action[0] = np.clip(-1,1,action[0])

        next_obs, reward, done, info = env.step(action)

        rgb = info['sensor.camera.rgb/front']
        segmentation = info['sensor.camera.semantic_segmentation/top']
        topdown = info['sensor.camera.rgb/top']

        # # load dense reward maps
        # positions = []
        # labels = []

        base_transform = ego_actor.get_transform()
        # base_waypoint = env.carla_interface.map.get_waypoint(base_transform.location, project_to_road=True)

        # pixel_x, pixel_y = np.meshgrid(np.arange(64), np.arange(64))
        # pixel_xy = np.stack([pixel_x.flatten(), pixel_y.flatten(), np.ones(64*64)], axis=-1)
        # world_pts = np.linalg.inv(calibration).dot(pixel_xy.T).T[:,:2]

        # # yaw = np.radians(((base_transform.rotation.yaw + 180) % 360) - 180)
        # yaw = -(((np.radians(base_transform.rotation.yaw) + np.pi) % (2*np.pi)) - np.pi)
        # rot_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        # world_pts = world_pts.dot(rot_matrix)

        # world_pts *= camera_actor.get_transform().location.z
        # world_pts[:,0] += camera_actor.get_transform().location.x
        # world_pts[:,1] += camera_actor.get_transform().location.y

        # for i, pt in enumerate(world_pts):
        #     x_loc, y_loc = pt[0], pt[1]
        #     location = carla.Location(x=x_loc, y=y_loc, z=base_transform.location.z)
        #     waypoint = env.carla_interface.map.get_waypoint(location, project_to_road=False)

        #     label = 0

        #     # check if off-road
        #     if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
        #         label = 1

        #     # check if lane violation
        #     else:
        #         if not waypoint.is_junction:
        #             base_yaw = base_waypoint.transform.rotation.yaw
        #             yaw = waypoint.transform.rotation.yaw
        #             waypoint_angle = (((base_yaw - yaw) + 180) % 360) - 180

        #             if np.abs(waypoint_angle) > 150:
        #                 label = 2

        #     positions.append((x_loc, y_loc))
        #     labels.append(label)

        # positions = np.array(positions)
        # labels = np.array(labels)

        # # check for vehicle collisions
        actor_list = env.carla_interface.actor_fleet.actor_list 
        actors = [actor for actor in actor_list
            if 'vehicle' in actor.type_id 
            and actor.get_transform().location.distance(base_transform.location) < 50
            and actor != ego_actor]

        # if len(actors) > 0:
        # bounding_boxes = [[(actor.bounding_box.extent.x, actor.bounding_box.extent.y),
        #                 (actor.bounding_box.extent.x, -actor.bounding_box.extent.y),
        #                 (-actor.bounding_box.extent.x, -actor.bounding_box.extent.y),
        #                 (-actor.bounding_box.extent.x, actor.bounding_box.extent.y)] for actor in actors]
        vehicles = [(actor.get_transform().location.x, actor.get_transform().location.y) for actor in actors]

        # bounding_boxes = np.array(bounding_boxes)
        vehicles = np.array(vehicles)
        #     num_vehicles = len(vehicles)

        #     for i in range(len(actors)):
        #         yaw = actors[i].get_transform().rotation.yaw
        #         bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)

        #     vehicles = bounding_boxes + vehicles[:,None,:]
        #     points = [Point(positions[i,0], positions[i,1]) for i in range(len(positions))]
        #     mask = np.zeros(len(labels))

        #     for i in range(len(actors)):
        #         poly = Polygon([(vehicles[i,j,0], vehicles[i,j,1]) for j in range(4)])
        #         in_poly = np.array([point.within(poly) for point in points])
        #         mask = np.logical_or(mask, in_poly)

        #     labels[mask] = 3

        # reward_map = np.zeros((64,64))
        # reward_map[pixel_xy[:,0].astype(int), pixel_xy[:,1].astype(int)] = labels
        # reward_map = reward_map[::-1]
        reward_map = None

        keys = list(info.keys())
        for key in keys:
            if sys.getsizeof(info[key]) > 2500:
                del info[key]
            elif isinstance(info[key], np.ndarray):
                info[key] = info[key].tolist()

        experience = {
            'obs': obs.tolist(),
            'next_obs': next_obs.tolist(),
            'action': action.tolist(),
            'reward': reward,
            'done': done.item(),

            'vehicles': vehicles.tolist()

            # 'actor_tf': transform_to_list(ego_actor.get_transform()),
            # 'camera_tf': transform_to_list(camera_actor.get_transform()),
        }
        experience.update(info)

        # np.save(os.path.join(save_path, 'world', '{:04d}'.format(step)), world_pts)
        save_env_state(rgb, segmentation, topdown, reward_map, experience, save_path, step)

        if done:
            print(step)
            break

        obs = next_obs

    return step + 1


def save_env_state(rgb, segmentation, topdown, reward_map, measurements, save_path, idx):
    rgb_path = os.path.join(save_path, 'rgb', '{:04d}.png'.format(idx))
    cv2.imwrite(rgb_path, rgb)

    segmentation_path = os.path.join(save_path, 'segmentation', '{:04d}.png'.format(idx))
    segmentation = segmentation.argmax(axis=-1)
    cv2.imwrite(segmentation_path, segmentation)

    topdown_path = os.path.join(save_path, 'topdown', '{:04d}.png'.format(idx))
    cv2.imwrite(topdown_path, topdown)

    # reward_path = os.path.join(save_path, 'reward', '{:04d}.png'.format(idx))
    # cv2.imwrite(reward_path, reward_map)

    measurements_path = os.path.join(save_path, 'measurements', '{:04d}.json'.format(idx))
    with open(measurements_path, 'w') as out:
        json.dump(measurements, out)


def transform_to_list(transform):
    location, rotation = transform.location, transform.rotation
    out = [location.x, location.y, location.z, rotation.pitch, rotation.yaw, rotation.roll]
    return out


def main(args):
    config = DefaultMainConfig()

    obs_config = LowDimObservationConfig()
    obs_config.sensors['sensor.camera.rgb/top'] = {
        'x':0.0,
        'z':18.0,
        'pitch':270,
        'sensor_x_res':'128',
        'sensor_y_res':'128',
        'fov':'120', \
        'sensor_tick': '0.0'}
    obs_config.sensors['sensor.camera.semantic_segmentation/top'] = {
        'x':0.0,
        'z':18.0,
        'pitch':270.0,
        'sensor_x_res': '128',
        'sensor_y_res':'128',
        'fov':'120',
        'sensor_tick': '0.0',
        'num_classes':5}

    scenario_config = NoCrashDenseTown01Config() # LeaderboardConfig()
    scenario_config.city_name = args.town
    # scenario_config.num_pedestrians = 250
    scenario_config.sample_npc = True
    scenario_config.num_npc_lower_threshold = 50
    scenario_config.num_npc_upper_threshold = 150
    scenario_config.use_scenarios = False

    action_config = MergedSpeedScaledTanhConfig()
    action_config.frame_skip = 5

    config.populate_config(
        observation_config=obs_config, 
        action_config=action_config, 
        scenario_config=scenario_config)
    config.server_fps = 20
    config.carla_gpu = args.gpu

    env = CarlaEnv(config=config, log_dir=args.path + '/')
    try:
        total_samples = 0
        while total_samples < args.n_samples:
            traj_length = collect_trajectory(env, args.path, args.speed)
            total_samples += traj_length
        print('Collected {} samples'.format(total_samples))
    finally:
        env.close()
        print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=500000)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--town', type=str, default='Town01')
    parser.add_argument('--path', type=str)
    parser.add_argument('--use_autopilot', type=int, default=1)
    parser.add_argument('--apply_noise', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    main(args)
