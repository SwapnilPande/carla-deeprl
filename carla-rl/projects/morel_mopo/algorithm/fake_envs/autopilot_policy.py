import projects.morel_mopo.algorithm.fake_envs.fake_env_utils as feutils

import numpy as np


class AutopilotPolicy():
    def __init__(self, env):
        self.lateral_controller = feutils.PIDLateralController(
                                        K_P=self.args_lateral_dict['K_P'],
                                        K_D=self.args_lateral_dict['K_D'],
                                        K_I=self.args_lateral_dict['K_I'],
                                        dt=self.args_lateral_dict['dt']
                                )

    def predict(self, obs):
        return self.get_autopilot_action(obs)


    def get_autopilot_action(self, obs, target_speed=0.5):

        angle  = obs[0]
        obstacle_dist = obs[4]

        steer = self.lateral_controller.pid_control(self.vehicle_pose.cpu().numpy(), angle)
        steer = np.clip(steer, -1, 1)


        if(obstacle_dist < 0.3):
            target_speed = -1.0

        return np.array([steer, target_speed])