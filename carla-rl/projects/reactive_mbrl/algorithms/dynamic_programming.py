import numpy as np
import torch
import time

from scipy.interpolate import RegularGridInterpolator
from projects.reactive_mbrl.data.reward_map import MAP_SIZE

ACTIONS = np.array(
    [
        [-1, 1 / 3],
        [-1, 2 / 3],
        [-1, 1],
        [-0.75, 1 / 3],
        [-0.75, 2 / 3],
        [-0.75, 1],
        [-0.5, 1 / 3],
        [-0.5, 2 / 3],
        [-0.5, 1],
        [-0.25, 1 / 3],
        [-0.25, 2 / 3],
        [-0.25, 1],
        [0, 1 / 3],
        [0, 2 / 3],
        [0, 1],
        [0.25, 1 / 3],
        [0.25, 2 / 3],
        [0.25, 1],
        [0.5, 1 / 3],
        [0.5, 2 / 3],
        [0.5, 1],
        [0.75, 1 / 3],
        [0.75, 2 / 3],
        [0.75, 1],
        [1, 1 / 3],
        [1, 2 / 3],
        [1, 1],
        [0, -1],
    ],
    dtype=np.float32,
).reshape(28, 2)
YAWS = np.linspace(-1.0, 1.0, 5)
SPEEDS = np.linspace(0, 8, 4)

num_yaws = len(YAWS)
num_spds = len(SPEEDS)
num_acts = len(ACTIONS)


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


class QSolver:
    def __init__(self, model):
        self.model = model

    def run_bellman(self, rewards, locs, next_locs, prev_V, discount_factor=0.9):
        with torch.no_grad():
            start = time.time()
            # Why?
            reward = (rewards == 0).float() - 1

            speeds, yaws, actions = (
                torch.tensor(SPEEDS),
                torch.tensor(YAWS),
                torch.tensor(ACTIONS),
            )

            grid_interpolator, theta, offset = initialize_grid_interpolator(
                next_locs, prev_V, speeds, yaws
            )

            Q = torch.zeros((MAP_SIZE, MAP_SIZE, num_spds, num_yaws, num_acts))

            for s, speed, in enumerate(speeds):
                for y, yaw in enumerate(yaws):
                    pred_locs, pred_yaws, pred_spds = self.predict(
                        locs, yaw, speed, actions
                    )

                    # convert locs to normalized grid coordinates and interpolate next Vs
                    next_Vs = interpolate(
                        grid_interpolator,
                        pred_locs,
                        pred_spds,
                        pred_yaws,
                        offset,
                        theta,
                    )

                    # Bellman backup
                    Q_target = bellman_backup(
                        next_Vs, reward, num_acts, discount_factor
                    )
                    Q[:, :, s, y] = torch.clamp(Q_target, -1, 0)

            # max over all actions
            V = Q.max(dim=-1)[0]
            V = V.detach().numpy()

        end = time.time()
        # print('total: {}'.format(end - start))
        return V, Q

    def solve(self, data):
        start = time.time()
        for i, (reward, world_pts, value_path, action_val_path) in enumerate(data):
            reward = reward.reshape(MAP_SIZE, MAP_SIZE)
            if i == 0:
                V = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1).repeat(
                    1, 1, num_spds, num_yaws
                )
                Q = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1, 1).repeat(
                    1, 1, num_spds, num_yaws, num_acts
                )
            else:
                V, Q = self.run_bellman(reward, world_pts, next_world_pts, V)

            np.save(value_path, V)
            np.save(action_val_path, Q)
            next_world_pts = world_pts
        end = time.time()
        print("total: {}".format(end - start))

    def predict(self, locs, yaw, speed, actions):
        pred_locs, pred_yaws, pred_spds = self.model.forward(
            locs[:, None, :].repeat(1, num_acts, 1).reshape(-1, 2),
            yaw[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
            speed[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
            actions[None].repeat(MAP_SIZE * MAP_SIZE, 1, 1).reshape(-1, 2),
        )

        return pred_locs, pred_yaws, pred_spds


def bellman_backup(next_Vs, reward, num_acts, discount_factor=0.9):
    terminal = (reward < 0)[..., None]
    #terminal = (reward > 0)[..., None]
    Q_target = reward[..., None] + (
        discount_factor * ~terminal * next_Vs.reshape(MAP_SIZE, MAP_SIZE, num_acts)
    )
    Q_target[torch.isnan(Q_target)] = -1
    return Q_target


def initialize_grid_interpolator(next_locs, prev_V, speeds, yaws):
    # normalize grid so we can use grid interpolation
    offset = next_locs[0]
    _next_locs = next_locs - offset
    theta = np.arctan2(_next_locs[-1][1], _next_locs[-1][0])
    _next_locs = rotate_pts(_next_locs, (np.pi / 4) - theta)

    min_x, min_y = np.min(_next_locs, axis=0)
    max_x, max_y = np.max(_next_locs, axis=0)

    # set up grid interpolator
    xs, ys = np.linspace(min_x, max_x, MAP_SIZE), np.linspace(min_y, max_y, MAP_SIZE)
    values = np.array(prev_V)
    # because indexing=ij, for more: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    values = np.moveaxis(values, 0, 1)
    grid_interpolator = RegularGridInterpolator(
        (xs, ys, speeds, yaws), values, bounds_error=False, fill_value=None
    )

    return grid_interpolator, theta, offset


def interpolate(grid_interpolator, pred_locs, pred_spds, pred_yaws, offset, theta):
    _pred_locs = pred_locs - offset
    _pred_locs = rotate_pts(_pred_locs, (np.pi / 4) - theta)
    pred_pts = np.concatenate([_pred_locs, pred_spds, pred_yaws], axis=1)
    next_Vs = grid_interpolator(pred_pts, method="linear")
    return next_Vs
