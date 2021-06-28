""" Utility functions """

import math

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18
import torchvision.transforms.functional as TF
import carla

# from carla_project.src.common import COLOR, CONVERTER


# Reward scaling parameters
LIGHT_COEFFICIENT = 50
COLLISION_COEFFICIENT = 50
STOP_SIGN_COEFFICIENT = 0
LANE_INVASION_COEFFICIENT = 50

# Reduce segmentation classes from 26
SEGMENTATION_MAP = np.uint8([
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    1,    # ped
    0,    # pole
    2,    # road line
    3,    # road
    4,    # sidewalk
    0,    # vegetation
    5,    # car
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # railtrack
    0,    # guardrail
    0,    # trafficlight
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    0,
    0,
    0
])

CONVERTER = np.uint8([
    0,    # unlabeled
    0,    # building
    0,    # fence
    0,    # other
    1,    # ped
    0,    # pole
    2,    # road line
    3,    # road
    4,    # sidewalk
    0,    # vegetation
    5,    # car
    0,    # wall
    0,    # traffic sign
    0,    # sky
    0,    # ground
    0,    # bridge
    0,    # railtrack
    0,    # guardrail
    0,    # trafficlight
    0,    # static
    0,    # dynamic
    0,    # water
    0,    # terrain
    6,
    7,
    8,
    ])


COLOR = np.uint8([
        (  0,   0,   0),    # unlabeled
        (220,  20,  60),    # ped
        (157, 234,  50),    # road line
        (128,  64, 128),    # road
        (244,  35, 232),    # sidewalk
        (  0,   0, 142),    # car
        (255,   0,   0),    # red light
        (255, 255,   0),    # yellow light
        (  0, 255,   0),    # green light
        ])


N_CLASSES = 6

# Image dimensions
RGB_IMAGE_SIZE = (64,64)
TOPDOWN_IMAGE_SIZE = (64,64)

# CALIBRATION = np.array([[128.,   0., 128.],
#                         [  0., 128.,  72.],
#                         [  0.,   0.,   1.]])

CALIBRATION = np.array([[548.99,   0., 256.],
                        [  0., 548.99,  256.],
                        [  0.,   0.,   1.]])

# CALIBRATION = np.array([[1097.987543300894,   0., 512.],
#                         [  0., 1097.987543300894,  512.],
#                         [  0.,   0.,   1.]])


def preprocess_rgb(img, grayscale=False, image_size=RGB_IMAGE_SIZE):
    """Preprocess RGB image

    Resizes image and converts to normalized tensor.

    Parameters
    ----------
    img : numpy.ndarray
        An RGB image of size (H, W, 3)

    Returns
    -------
    torch.Tensor
        Processed image of size (3, H, W) or (1, H, W)
    """
    img = TF.to_pil_image(img)
    img = TF.resize(img, image_size)
    if grayscale:
        img = TF.to_grayscale(img)
    img = TF.to_tensor(img)
    return img


def preprocess_topdown(img, crop=True):
    """Preprocess topdown segmentation map

    Reduces classes, resizes image, and converts to one-hot tensor.

    Parameters
    ----------
    img : numpy.ndarray
        A segmentation map of size (H, W, 3) or (H, W)

    Returns
    -------
    torch.Tensor
        Processed one-hot image of size (NUM_CLASSES, H, W)
    """
    if len(img.shape) > 2:
        img = img[:,:,0]

    # img = COLOR[CONVERTER[img]]
    img = SEGMENTATION_MAP[img]
    if crop:
        img = TF.to_pil_image(img)
        img = TF.center_crop(img, 192)
        img = TF.resize(img, TOPDOWN_IMAGE_SIZE)
    img = (TF.to_tensor(img) * 255).long()[0]
    img = torch.nn.functional.one_hot(img, N_CLASSES).permute(2,0,1).float()
    return img


def get_reward(curr, return_is_terminal=False):
    """Compute reward given current episode measurements

    Parameters
    ----------
    curr : Dictionary of episode measurements
    """

    speed_reward = curr['speed']
    pos = curr['x'], curr['y']
    wp0, wp1 = (curr['near_x0'], curr['near_y0']), (curr['near_x1'], curr['near_y1'])
    dist_to_traj = distance_from_point_to_line(pos, wp0, wp1)
    dist_reward = -np.abs(dist_to_traj)

    runover_light = curr['red_light_test'] == "FAILURE"
    is_collision = curr['collision_test'] == "FAILURE"
    runover_stop = curr['stop_test'] == "FAILURE"
    lane_invasion = curr['lane_test'] == "FAILURE"

    penalties = (runover_light * -LIGHT_COEFFICIENT) + (is_collision * -COLLISION_COEFFICIENT) + \
        (runover_stop * -STOP_SIGN_COEFFICIENT) + (lane_invasion * -LANE_INVASION_COEFFICIENT)

    reward = (dist_reward + speed_reward + penalties) / 16.

    if return_is_terminal:
        return reward, int(penalties < 0.)
    else:
        return reward


def get_action(curr):
    """ Recover action from measurements. """
    steer = get_angle_to_next_node(curr) / 180
    gas = np.clip(curr['target_speed'] / 10., -1., 1.) if not curr['brake'] else -.7
    return np.array([steer, gas])


def get_obs(sample):
    """ Construct observation array from sample dict """
    next_orientation = get_angle_to_next_node(sample) / 180.
    speed = sample['speed'] / 10.
    steer = 0 # sample['steer']

    pos = sample['x'], sample['y']
    wp0, wp1 = (sample['near_x0'], sample['near_y0']), (sample['near_x1'], sample['near_y1'])
    dist_to_traj = distance_from_point_to_line(pos, wp0, wp1) / 10.

    nearby_actors = sample['nearby_actors']
    nearby_actors = [actor for actor in nearby_actors if actor['dist'] < 10.]
    nearby_actors = sorted(nearby_actors, key=lambda x: x['dist'])
    if len(nearby_actors) > 0:
        nearest = nearby_actors[0]
        is_obstacle = 1.
        obstacle_dist = nearest['dist'] / 10.
        obstacle_theta = np.arctan2(nearest['vel'][1], nearest['vel'][0]) / (2*np.pi)
        obstacle_vel = np.linalg.norm(nearest['vel']) / 10.
    else:
        is_obstacle = 0.
        obstacle_dist = 1.
        obstacle_theta = 0.
        obstacle_vel = 0.

    is_light = sample['light']
    ldist = sample['ldist'] / 10.

    is_walker = sample['walker']
    wdist = sample['wdist'] / 10.

    is_vehicle = sample['vehicle']
    vdist = sample['vdist'] / 10.

    mlp_features = np.array([next_orientation, speed, steer, dist_to_traj, \
        is_light, ldist, is_obstacle, obstacle_dist, obstacle_theta, obstacle_vel, \
        is_walker, wdist, is_vehicle, vdist])

    return mlp_features


def get_angle_to_next_node(curr):
    """ Compute angle to next node from measurements. """

    pos = np.array([curr['x'], curr['y']])
    target = np.array([curr['near_x1'], curr['near_y1']])
    theta = curr['theta']
    return get_angle_to(pos, theta, target)


def get_angle_to(pos, theta, target):
    """ Compute angle to target from current position and orientation. """

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle 

    return angle


def get_dir(path):
    return '/'.join(path.split('/')[:-1])


def distance_from_point_to_line(pt, l1, l2):
    pt = np.array([pt[0], pt[1]])
    l1 = np.array([l1[0], l1[1]])
    l2 = np.array([l2[0], l2[1]])

    a_vec = l2 - l1
    b_vec = pt - l1
    dist = np.cross(a_vec, b_vec) / np.linalg.norm(a_vec)
    if not np.isfinite(dist):
        dist = 0
    return dist


class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes.
    Copied from CARLA PythonAPI examples.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, sensor_transform):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, sensor_transform) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_bounding_box(vehicle, sensor_transform):
        """
        Returns 3D bounding box for a vehicle based on sensor_transform.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, sensor_transform)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(CALIBRATION, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor_transform):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor_transform)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor_transform):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor_transform)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def rotate(point, angle, origin=(256,256)):
    """ https://stackoverflow.com/a/34374437/14993919 """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
