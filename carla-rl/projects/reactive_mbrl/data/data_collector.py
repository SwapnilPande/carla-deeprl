import datetime
import os
import numpy as np
import cv2
import json

from projects.reactive_mbrl.data.reward_map import (
    calculate_reward_map, calculate_world_grid_points,
)

WIDE_CROP_TOP = 48
NARROW_CROP_BOTTOM = 80


class DataCollector:

    def __init__(self, env, path):
        self.env = env
        self.path = path
        self.waypointer = None

    def collect_trajectory(self, speed, max_path_length=5000):
        print(f"Collecting a new trajectory.")
        self.setup_output_dir()
        obs = self.env.reset()

        for step in range(max_path_length):
            expert_action = self.env.get_autopilot_action(speed)
            next_obs, reward, done, info = self.env.step(expert_action)

            reward_map, world_pts = calculate_reward_map(self.env)
            experience = create_experience(obs, next_obs, expert_action, done, reward, info)
            self.save_data(reward_map, world_pts, experience, info, step)
            if done:
                break

            obs = next_obs
        print(f"Done collecting trajectory, number of steps {step + 1}")
        return step + 1

    def setup_output_dir(self):
        now = datetime.datetime.now()
        salt = np.random.randint(100)
        fname = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second, salt)))
        self.save_path = os.path.join(self.path, fname)
        print(f"Setting up output directory {self.save_path}")

        # check for conflicts
        if os.path.isdir(self.save_path):
            print('Directory conflict, trying again...')
            return 0

        os.mkdir(self.save_path)
        os.mkdir(os.path.join(self.save_path, 'topdown'))
        os.mkdir(os.path.join(self.save_path, 'wide_rgb'))
        os.mkdir(os.path.join(self.save_path, 'wide_seg'))
        os.mkdir(os.path.join(self.save_path, 'narrow_rgb'))
        os.mkdir(os.path.join(self.save_path, 'narrow_seg'))
        os.mkdir(os.path.join(self.save_path, 'reward'))
        os.mkdir(os.path.join(self.save_path, 'world'))
        os.mkdir(os.path.join(self.save_path, 'measurements'))


    def save_data(self, reward_map, world_pts, experience, info, idx):
        cropped_wide_rgb = info['sensor.camera.rgb/front_wide'][WIDE_CROP_TOP:,:,::-1]
        cropped_wide_seg = info['sensor.camera.semantic_segmentation/front_wide'][WIDE_CROP_TOP:]
        cropped_narr_rgb = info['sensor.camera.rgb/front_narrow'][:-NARROW_CROP_BOTTOM,:,::-1]
        cropped_narr_seg = info['sensor.camera.semantic_segmentation/front_narrow'][:-NARROW_CROP_BOTTOM]
        self.save_img(cropped_wide_rgb, 'wide_rgb', idx)
        self.save_img(cropped_wide_seg, 'wide_seg', idx)
        self.save_img(cropped_narr_rgb, 'narrow_rgb', idx)
        self.save_img(cropped_narr_seg, 'narrow_seg', idx)
        self.save_img(info['sensor.camera.rgb/top'], 'topdown', idx)

        cmd = self.env.carla_interface.next_cmds[0]
        experience['cmd_name'] = cmd.name
        experience['cmd_value'] = cmd.value

        np.save(os.path.join(self.save_path, 'reward', '{:04d}'.format(idx)), reward_map)
        np.save(os.path.join(self.save_path, 'world', '{:04d}'.format(idx)), world_pts)
        measurements_path = os.path.join(self.save_path, 'measurements', '{:04d}.json'.format(idx))
        with open(measurements_path, 'w') as out:
            json.dump(experience, out)

    def save_img(self, img_arr, name, idx):
        img_path = os.path.join(self.save_path, name, '{:04d}.png'.format(idx))
        cv2.imwrite(img_path, img_arr)


def create_experience(obs, next_obs, action, done, reward, info):
    return {
        'obs': obs.tolist(),
        'next_obs': next_obs.tolist(),
        'action': action.tolist(),
        'reward': reward,
        'done': done.item(),
        'speed': info['speed'],
        'gnss': info['sensor.other.gnss']
    }
