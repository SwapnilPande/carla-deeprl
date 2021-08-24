import numpy as np
import torch
import time
import sys
import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from projects.reactive_mbrl.data.reward_map import MAP_SIZE, MIN_REWARD, MAX_LOSS

# ACTIONS = np.array([[-1, 1 / 3], [0, 1 / 3], [1, 1 / 3],], dtype=np.float32).reshape(
#     3, 2
# )

steer = np.linspace(-1, 1, 101, dtype=np.float32)
throt = np.linspace(0.0, 1, 2, endpoint=False, dtype=np.float32)
num_steer = len(steer)
num_throt = len(throt)
ACTIONS = np.transpose([np.tile(throt, len(steer)), np.repeat(steer, len(throt))])
ACTIONS = np.copy(ACTIONS[:, ::-1])
ACTIONS = np.insert(ACTIONS, len(ACTIONS), [0.0, -20.0], axis=0)

# ACTIONS = np.array(
#     [
#         [-1, 1 / 3],
#         [-1, 2 / 3],
#         [-1, 1],
#         [-0.75, 1 / 3],
#         [-0.75, 2 / 3],
#         [-0.75, 1],
#         [-0.5, 1 / 3],
#         [-0.5, 2 / 3],
#         [-0.5, 1],
#         [-0.25, 1 / 3],
#         [-0.25, 2 / 3],
#         [-0.25, 1],
#         [0, 1 / 3],
#         [0, 2 / 3],
#         [0, 1],
#         [0.25, 1 / 3],
#         [0.25, 2 / 3],
#         [0.25, 1],
#         [0.5, 1 / 3],
#         [0.5, 2 / 3],
#         [0.5, 1],
#         [0.75, 1 / 3],
#         [0.75, 2 / 3],
#         [0.75, 1],
#         [1, 1 / 3],
#         [1, 2 / 3],
#         [1, 1],
#         # [0, -1],
#     ],
#     dtype=np.float32,
# ).reshape(27, 2)
STEERINGS = np.array([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 0])
THROTS = np.array([1 / 3, 2 / 3, 1, -1])
YAWS = np.linspace(-1.0, 1.0, 5)
SPEEDS = np.linspace(0, 8, 4)

NORMALIZING_ANGLE = np.pi / 4

num_yaws = len(YAWS)
num_spds = len(SPEEDS)
num_acts = len(ACTIONS)


def rotate_pts(pts, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R.dot(pts.T).T


class OnlineQSolver:
    def __init__(self, model):
        self.model = model

    def predict(self, locs, yaw, speed, actions):
        pred_locs, pred_yaws, pred_spds = self.model.forward(
            locs[:, None, :].repeat(1, num_acts, 1).reshape(-1, 2),
            yaw[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
            speed[None, None].repeat(MAP_SIZE * MAP_SIZE, num_acts, 1).reshape(-1, 1),
            actions[None].repeat(MAP_SIZE * MAP_SIZE, 1, 1).reshape(-1, 2),
        )

        return pred_locs, pred_yaws, pred_spds

    def solve(self, V, locs, yaw, speed):
        V = np.transpose(V, (1, 0))
        
        locs = torch.tensor(locs)
        V = torch.tensor(V)
        yaw = torch.tensor(yaw)
        speed = torch.tensor(speed)

        for _ in range(1):
            V, Q = self.run_bellman(V, locs, yaw, speed)
        return V, Q

    def run_bellman(self, prev_V, locs, ref_yaw, ref_speed, discount_factor=0.9):
        with torch.no_grad():
            start = time.time()
            # reward = (rewards == 0).float() - 1

            actions = torch.tensor(ACTIONS)

            grid_interpolator, theta, offset = initialize_grid_interpolator(
                locs, prev_V
            )

            Q = torch.zeros((MAP_SIZE, MAP_SIZE, num_acts))
            terminal = (prev_V >= MAX_LOSS)[..., None]

            pred_locs, pred_yaws, pred_spds = self.predict(
                locs, ref_yaw, ref_speed, actions
            )

            # convert locs to normalized grid coordinates and interpolate next Vs
            next_Vs = interpolate(grid_interpolator, pred_locs, offset, theta)

            # Bellman backup
            Q_target = bellman_backup(next_Vs, prev_V, terminal, discount_factor)
            Q[:, :] = torch.clamp(Q_target, 0, MAX_LOSS)
            # Incure a small lost for stopping.
            Q[:, :, -1] += 0.1

            # max over all actions
            V = Q.max(dim=-1)[0]

        end = time.time()
        # print('total: {}'.format(end - start))
        return V, Q


class QSolver:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def run_bellman(
        self, rewards, locs, next_locs, prev_V, measurement, discount_factor=0.9
    ):
        with torch.no_grad():
            start = time.time()
            # reward = (rewards == 0).float() - 1
            ref_yaw = torch.tensor(np.radians(measurement["yaw"]))

            speeds, yaws, actions = (
                torch.tensor(SPEEDS),
                torch.tensor(YAWS),
                torch.tensor(ACTIONS),
            )

            grid_interpolator, theta, offset = initialize_grid_interpolator(
                next_locs, prev_V, speeds, yaws
            )

            Q = torch.zeros((MAP_SIZE, MAP_SIZE, num_spds, num_yaws, num_acts))
            terminal = (rewards <= MIN_REWARD)[..., None]
            action_rewards = self.calculate_action_rewards(ACTIONS)
            final_reward = rewards[..., None] + action_rewards
            # final_reward = rewards[..., None]

            for s, speed, in enumerate(speeds):
                for y, yaw in enumerate(yaws):
                    pred_locs, pred_yaws, pred_spds = self.predict(
                        locs, ref_yaw + yaw, speed, actions
                    )

                    # convert locs to normalized grid coordinates and interpolate next Vs
                    next_Vs = interpolate(
                        grid_interpolator,
                        pred_locs,
                        pred_spds,
                        pred_yaws - ref_yaw,
                        offset,
                        theta,
                    )
                    if speed == 8 and yaw == 0.0:
                        record_pred_locs = pred_locs

                    # Bellman backup
                    Q_target = bellman_backup(
                        next_Vs, final_reward, terminal, discount_factor
                    )
                    Q[:, :, s, y] = torch.clamp(Q_target, MIN_REWARD, 0)

            # max over all actions
            V = Q.max(dim=-1)[0]

        end = time.time()
        # print('total: {}'.format(end - start))
        return V, Q, record_pred_locs

    def solve(self, data):
        start = time.time()
        for (
            i,
            (
                reward,
                world_pts,
                value_path,
                action_val_path,
                next_preds_path,
                measurement,
            ),
        ) in enumerate(data):
            reward = reward.reshape(MAP_SIZE, MAP_SIZE)
            reward = np.transpose(reward, (1, 0))
            if i == 0:
                V = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1).repeat(
                    1, 1, num_spds, num_yaws
                )
                Q = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1, 1).repeat(
                    1, 1, num_spds, num_yaws, num_acts
                )
                next_preds = world_pts
            else:
                V, Q, next_preds = self.run_bellman(
                    reward, world_pts, next_world_pts, V, measurement
                )

            np.save(value_path, V)
            np.save(action_val_path, Q)
            np.save(next_preds_path, next_preds)
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

    def calculate_action_rewards(self, actions):
        steering = actions[:, 0]
        steering_penalty = -np.abs(steering) * float(self.config.steering_reward_coeff)
        steering_penalty = torch.tensor(steering_penalty)
        return (
            steering_penalty.reshape(num_acts, 1)
            .repeat(1, MAP_SIZE * MAP_SIZE)
            .reshape(num_acts, MAP_SIZE, MAP_SIZE)
            .transpose(0, 2)
        )


def bellman_backup(next_Vs, current_V, terminal, discount_factor=0.9):
    Q_target = current_V[..., None] + (
        discount_factor * ~terminal * next_Vs.reshape(MAP_SIZE, MAP_SIZE, num_acts)
    )
    Q_target[torch.isnan(Q_target)] = MAX_LOSS
    return Q_target


def initialize_grid_interpolator(next_locs, prev_V):
    # normalize grid so we can use grid interpolation
    offset = next_locs[0]
    _next_locs = next_locs - offset
    theta = np.arctan2(_next_locs[-1][1], _next_locs[-1][0])
    _next_locs = rotate_pts(_next_locs, NORMALIZING_ANGLE - theta)

    min_x, min_y = np.min(_next_locs, axis=0)
    max_x, max_y = np.max(_next_locs, axis=0)

    # set up grid interpolator
    xs, ys = np.linspace(min_x, max_x, MAP_SIZE), np.linspace(min_y, max_y, MAP_SIZE)
    values = np.array(prev_V)
    # because indexing=ij, for more: https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    values = np.moveaxis(values, 0, 1)
    grid_interpolator = RegularGridInterpolator(
        (xs, ys), values, bounds_error=False, fill_value=None
    )

    return grid_interpolator, theta, offset


def interpolate(grid_interpolator, pred_locs, offset, theta):
    _pred_locs = pred_locs - offset
    _pred_locs = rotate_pts(_pred_locs, NORMALIZING_ANGLE - theta)
    # pred_pts = np.concatenate([_pred_locs], axis=1)
    next_Vs = grid_interpolator(_pred_locs, method="linear")
    return next_Vs
