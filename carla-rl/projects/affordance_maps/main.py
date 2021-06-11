import traceback
import math

from shapely.geometry import Point, Polygon

import numpy as np
import cv2
import matplotlib.pyplot as plt
import carla
from client_bounding_boxes import ClientSideBoundingBoxes
from environment import CarlaEnv
from environment.config.config import DefaultMainConfig
from environment.config.observation_configs import *

np.random.seed(1)
plt.ion()

config = DefaultMainConfig()
obs_config = PerspectiveRGBObservationConfig() # LowDimObservationConfig()
obs_config.sensors['sensor.camera.rgb/top'] = {
    'x':13.0,
    'z':18.0,
    'pitch':270,
    'sensor_x_res':'64',
    'sensor_y_res':'64',
    'fov':'90', \
    'sensor_tick': '0.0'}
config.populate_config(observation_config=obs_config)
env = CarlaEnv(config=config)

def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])

try:
    env.reset()
    calibration = np.array([[32, 0, 32],
                            [0, 32, 32],
                            [0,  0,  1]])
    ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
    camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/top'].sensor
    camera_actor.calibration = calibration

    for i in range(2000):
        action = env.get_autopilot_action()
        # action = np.array([0,-1])
        obs, _, done, _ = env.step(action)
        image = env.render()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get waypoint annotations
        # annotations = []
        positions = []
        labels = []

        base_transform = ego_actor.get_transform()
        base_waypoint = env.carla_interface.map.get_waypoint(base_transform.location, project_to_road=True)

        pixel_x, pixel_y = np.meshgrid(np.arange(64), np.arange(64))
        pixel_xy = np.stack([pixel_x.flatten(), pixel_y.flatten(), np.ones(64*64)], axis=-1)
        world_pts = np.linalg.inv(calibration).dot(pixel_xy.T).T * camera_actor.get_transform().location.z
        world_pts[:,0] += camera_actor.get_transform().location.x
        world_pts[:,1] += camera_actor.get_transform().location.y

        for i, pt in enumerate(world_pts):
            x_loc, y_loc = pt[0], pt[1]
            location = carla.Location(x=x_loc, y=y_loc, z=base_transform.location.z)
            waypoint = env.carla_interface.map.get_waypoint(location, project_to_road=False)

            label = 0

            # check if off-road
            if waypoint is None or waypoint.lane_type != carla.LaneType.Driving:
                label = 1

            # check if lane violation
            else:
                if not waypoint.is_junction:
                    base_yaw = base_waypoint.transform.rotation.yaw
                    yaw = waypoint.transform.rotation.yaw
                    waypoint_angle = (((base_yaw - yaw) + 180) % 360) - 180

                    if np.abs(waypoint_angle) > 150:
                        label = 2

            positions.append((x_loc, y_loc))
            labels.append(label)

        positions = np.array(positions)
        labels = np.array(labels)

        # check for vehicle collisions
        actors = [actor for actor in env.carla_interface.actor_fleet.actor_list 
            if 'vehicle' in actor.type_id and actor.get_transform().location.distance(base_transform.location) < 30]

        bounding_boxes = [[(actor.bounding_box.extent.x, actor.bounding_box.extent.y),
                           (actor.bounding_box.extent.x, -actor.bounding_box.extent.y),
                           (-actor.bounding_box.extent.x, -actor.bounding_box.extent.y),
                           (-actor.bounding_box.extent.x, actor.bounding_box.extent.y)] for actor in actors]
        vehicles = [(actor.get_transform().location.x, actor.get_transform().location.y) for actor in actors]

        bounding_boxes = np.array(bounding_boxes)
        vehicles = np.array(vehicles)
        num_vehicles = len(vehicles)

        for i in range(len(actors)):
            yaw = actors[i].get_transform().rotation.yaw
            bounding_boxes[i] = rotate_points(bounding_boxes[i], yaw)

        vehicles = bounding_boxes + vehicles[:,None,:]
        points = [Point(positions[i,0], positions[i,1]) for i in range(len(positions))]
        mask = np.zeros(len(labels))

        for i in range(len(actors)):
            poly = Polygon([(vehicles[i,j,0], vehicles[i,j,1]) for j in range(4)])
            in_poly = np.array([point.within(poly) for point in points])
            mask = np.logical_or(mask, in_poly)

        labels[mask] = 3

        reward_map = np.zeros((64,64))
        reward_map[pixel_xy[:,0].astype(int), pixel_xy[:,1].astype(int)] = labels

        plt.clf()
        plt.imshow(reward_map)
        # plt.scatter(positions[:,0], positions[:,1], color=labels)
        # plt.scatter([base_transform.location.x], [base_transform.location.y], color='green')
        # plt.colorbar()
        plt.show(block=False)
        plt.pause(.01)

        # # get vehicle boxes
        # for vehicle_actor in env.carla_interface.actor_fleet.actor_list:
        #     if 'vehicle' not in vehicle_actor.type_id:
        #         continue

        #     box = ClientSideBoundingBoxes.get_bounding_box(vehicle_actor.get_transform(), camera_actor.get_transform(), calibration)
        #     box = box.astype(int)
        #     for i in range(1):
        #         cv2.circle(image, (box[i,0], box[i,1]), radius=1, color=(0,255,0), thickness=-1)

        cv2.imshow('pov', image)
        cv2.waitKey(1)

        if done:
            env.reset()
            ego_actor = env.carla_interface.get_ego_vehicle()._vehicle
            camera_actor = env.carla_interface.actor_fleet.sensor_manager.sensors['sensor.camera.rgb/top'].sensor
            camera_actor.calibration = calibration
except:
    traceback.print_exc()
    # import ipdb; ipdb.set_trace()
finally:
    env.close()
    cv2.destroyAllWindows()