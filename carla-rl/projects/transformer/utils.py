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

# CALIBRATION = np.array([[548.99,   0., 256.],
#                         [  0., 548.99,  256.],
#                         [  0.,   0.,   1.]])

# CALIBRATION = np.array([[1097.987543300894,   0., 512.],
#                         [  0., 1097.987543300894,  512.],
#                         [  0.,   0.,   1.]])

CALIBRATION = np.array([[32, 0, 32],
                        [0, 32, 32],
                        [0,  0,  1]])


def preprocess_rgb(img, grayscale=False):
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
    img = TF.resize(img, RGB_IMAGE_SIZE)
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
