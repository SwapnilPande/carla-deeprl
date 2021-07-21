import numpy as np
import torch
import time
import sys
import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from projects.reactive_mbrl.data.reward_map import MAP_SIZE, MIN_REWARD

# ACTIONS = np.array([[-1, 1 / 3], [0, 1 / 3], [1, 1 / 3],], dtype=np.float32).reshape(
#     3, 2
# )

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
            #final_reward = rewards[..., None]

            for s, speed, in enumerate(speeds):
                for y, yaw in enumerate(yaws):
                    # pred_locs, pred_yaws, pred_spds = self.predict(
                    #     locs, yaw, speed, actions
                    # )
                    pred_locs, pred_yaws, pred_spds = self.predict(
                        locs, ref_yaw + yaw, speed, actions
                    )

                    # convert locs to normalized grid coordinates and interpolate next Vs
                    next_Vs = interpolate(
                        grid_interpolator,
                        pred_locs,
                        pred_spds,
                        pred_yaws - ref_yaw,
                        # pred_yaws,
                        offset,
                        theta,
                    )
                    # ego_pos = np.array(
                    #     [measurement["ego_vehicle_x"], measurement["ego_vehicle_y"]]
                    # )
                    # plt.plot(ego_pos[0], ego_pos[1], "x", color="red", label="AV")
                    # plt.plot(
                    #     locs[:, 0],
                    #     locs[:, 1],
                    #     "o",
                    #     color="black",
                    #     alpha=0.5,
                    #     label="current",
                    # )
                    # plt.plot(
                    #     next_locs[:, 0],
                    #     next_locs[:, 1],
                    #     "o",
                    #     color="green",
                    #     label="next",
                    # )
                    # plt.plot(
                    #     pred_locs[:, 0], pred_locs[:, 1], "o", color="red", label="next"
                    # )
                    # plt.legend()
                    # plt.savefig("/zfsauton/datasets/ArgoRL/jhoang/logs/world.png")
                    # plt.clf()
                    # next_Vs = next_Vs.reshape(16, 16, num_acts)
                    # plt.pcolormesh(next_Vs[:, :, 0])
                    # plt.savefig("/zfsauton/datasets/ArgoRL/jhoang/logs/next_Vs_-1.png")
                    # plt.clf()
                    # plt.pcolormesh(next_Vs[:, :, 1])
                    # plt.savefig("/zfsauton/datasets/ArgoRL/jhoang/logs/next_Vs_0.png")
                    # plt.clf()
                    # plt.pcolormesh(next_Vs[:, :, 2])
                    # plt.savefig("/zfsauton/datasets/ArgoRL/jhoang/logs/next_Vs_1.png")
                    # plt.clf()
                    # sys.exit(0)

                    # Bellman backup
                    Q_target = bellman_backup(
                        next_Vs, final_reward, terminal, discount_factor
                    )
                    Q[:, :, s, y] = torch.clamp(Q_target, MIN_REWARD, 0)

            # max over all actions
            V = Q.max(dim=-1)[0]
            # max_action = torch.argmax(Q, dim=-1)[8, 8, 3, :]
            # print(ACTIONS[max_action])

        end = time.time()
        # print('total: {}'.format(end - start))
        return V, Q

    def solve(self, data):
        start = time.time()
        for (
            i,
            (reward, world_pts, value_path, action_val_path, measurement),
        ) in enumerate(data):
            reward = reward.reshape(MAP_SIZE, MAP_SIZE)
            if i == 0:
                V = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1).repeat(
                    1, 1, num_spds, num_yaws
                )
                Q = reward.reshape(MAP_SIZE, MAP_SIZE, 1, 1, 1).repeat(
                    1, 1, num_spds, num_yaws, num_acts
                )
            else:
                V, Q = self.run_bellman(
                    reward, world_pts, next_world_pts, V, measurement
                )

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


def bellman_backup(next_Vs, reward, terminal, discount_factor=0.9):
    Q_target = reward + (
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
