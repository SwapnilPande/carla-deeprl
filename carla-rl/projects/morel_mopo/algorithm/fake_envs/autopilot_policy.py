import projects.morel_mopo.algorithm.fake_envs.fake_env_utils as feutils

import numpy as np


class AutopilotPolicy():
    def __init__(self, env):
        self.lateral_controller = feutils.PIDLateralController(
                                        K_P = 0.88,
                                        K_D = 0.02,
                                        K_I = 0.5,
                                        dt  = 1/10.0
                                )
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    def predict(self, obs):
        return self.get_autopilot_action(obs)


    def get_autopilot_action(self, obs, target_speed=1.0):

        angle  = obs[...,0]
        obstacle_dist = obs[...,4]

        steer = self.lateral_controller.pid_control(angle)
        steer = np.clip(steer, -1, 1)


        if(obstacle_dist < 0.6):
            target_speed = -1.0

        return np.array([steer[0], target_speed])