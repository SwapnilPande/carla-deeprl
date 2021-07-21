import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import shutil
import argparse
from collections import deque
import copy

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
from projects.morel_mopo.algorithm.dynamics_models import GRUDynamicsEnsemble, ProbabilisticGRUDynamicsEnsemble
from projects.morel_mopo.algorithm.fake_envs import RNNFakeEnv
from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig

# Environment
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig






EXPERIMENT_NAME = "first_test"
TAGS = ["dyn_only"]


def rot(theta):
    R = np.array([[ np.cos(theta), -np.sin(theta)],
                      [ np.sin(theta), np.cos(theta)]])
    return R


############################### DEFAULT POLICIES ###############################
# Default policies to test dynamics models with
class RandomPolicy:
    def __init__(self, env):
        self.action_shape = env.action_space.shape

    def __call__(self, obs):
        return np.random.rand(*self.action_shape)


class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()

################################################################################

def generate_video(logger, image_path, save_path, name):
        vid_path = os.path.join(save_path, name + '.mp4')

        im_path = os.path.join(image_path, "%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path , "-framerate", '25', "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()


        logger.log_asset("dynamics_eval/videos", vid_path)

        # Clear the temporary directory of images after video is generated
        shutil.rmtree(image_path)
        os.mkdir(image_path)


def gen_state_pic(cur_state, next_state, ensemble_predictions, save_dir, idx):
    plt.figure()

    # First, plot the current car position
    x, y, theta = cur_state
    plt.plot(x, y, 'o', color = "blue")

    # Plot next state
    next_x, next_y, theta = next_state
    plt.plot(next_x, next_y, 'o', color = "red")

    # Plot dynamics next state predictions
    next_x_predictions = ensemble_predictions[:, 0]
    next_y_predictions = ensemble_predictions[:, 1]
    plt.plot(next_x_predictions, next_y_predictions, "x", color = "orange")

    # Center agent on the plot
    plt.xlim(x-5, x+5)
    plt.ylim(y-5, y+5)

    plt.savefig(os.path.join(save_dir, "{:04d}.png".format(idx)))
    plt.close()


def n_step_eval(exp_name, logger, real_env, fake_env, policy, num_episodes, n = 1, generate_videos = False):
    # List to store rollouts errors
    # Store each step separately
    n_step_rollout_squared_errors = np.zeros(shape = (n,))
    n_step_reward_squared_errors = np.zeros(shape = (n,))
    n_step_rollout_squared_errors = n_step_rollout_squared_errors
    n_step_rollout_uncertainty = []
    obs_dim = real_env.observation_space.shape[0]

    # Get the number of stacked frames fake_env needs
    # frame_stack = fake_env.frame_stack
    frame_stack = 10

    if(generate_videos):
        video_save_dir = logger.prep_dir("dynamics_eval/videos")
        image_save_dir = logger.prep_dir("dynamics_eval/temp")

    total_steps = 0
    for i in range(num_episodes):
        print("Experiment: {}, Episode {}".format(exp_name, i))
        real_dynamics_obs_history = deque([], maxlen = frame_stack)
        real_dynamics_action_history = deque([], maxlen = frame_stack)
        real_dynamics_pose = None

        # Reset environment and get real observation
        real_obs = real_env.reset()
        # Do this step because we need to data returned in info, which reset does not return
        real_obs, real_reward, real_done, real_info = real_env.step(np.array([0,0]))


        # Since the fake environment requires a frame stack input, we have to stack up the frames
        # Loop over frame stack to collect the correct rollout stack to start using the fake_env
        for frame in range(frame_stack):
            real_action = policy(real_obs)
            real_dynamics_action_history.append(real_action)

            real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))

            real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                             real_info["ego_vehicle_y"],
                             real_info["ego_vehicle_theta"]])

            real_obs, reward, done, real_info = real_env.step(real_action)

        real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))

        real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                             real_info["ego_vehicle_y"],
                             real_info["ego_vehicle_theta"]])



        fake_obs = fake_env.reset(inp = (
                    np.array(real_dynamics_obs_history),
                    np.array(real_dynamics_action_history),
                    real_dynamics_pose,
                    real_info["waypoints"]
                )
            )


        real_done = False
        real_rollout_steps = 0
        fake_rollout_steps = 0
        # import ipdb; ipdb.set_trace()
        while not real_done and real_rollout_steps < 1000:

            action = policy(real_obs)

            # Generate new observations


            real_next_obs, real_reward, real_done, real_info =  real_env.step(action)
            fake_next_obs, fake_reward, fake_done, fake_info =  fake_env.step(action)

            # Compare real obs, fake obs
            # Compare real reward, fake reward
            n_step_rollout_squared_errors[fake_rollout_steps] += (1/obs_dim) * np.sum(real_next_obs - fake_next_obs)**2
            n_step_reward_squared_errors[fake_rollout_steps] += (real_reward - fake_reward)**2


            # Generate a video of the state prediction
            #TODO GET ENSEMBLE PREDICTION

            if(generate_videos):
                next_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                                                real_info["ego_vehicle_y"],
                                                real_info["ego_vehicle_theta"]])

                delta = next_dynamics_pose - real_dynamics_pose

                transformed = np.linalg.inv(rot(np.deg2rad(real_dynamics_pose[2]))) @ np.expand_dims(delta[0:2], axis = -1)

                # print("REAL DELTA: {:.4f} {:.4f} {:.4f}".format(float(delta[0]), float(delta[1]), float(delta[2])))
                # print("REAL TRANSFORMED: {:.4f} {:.4f} {:.4f}".format(float(np.squeeze(transformed)[0]), float(np.squeeze(transformed)[1]), float(delta[2])))
                # if(delta[2] > 0.1):
                #     input()


                gen_state_pic(real_dynamics_pose, next_dynamics_pose, fake_info['predictions'], image_save_dir, real_rollout_steps)

            # print(real_dynamics_pose)
            # print(fake_info['predictions'])
            # print()

            real_obs = real_next_obs
            fake_obs = fake_next_obs
            real_dynamics_pose = next_dynamics_pose
            real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))
            real_dynamics_action_history.append(action)



            # Increment rollout counters
            real_rollout_steps += 1
            fake_rollout_steps += 1
            total_steps += 1

            # If we have reached desired fake env rollout length
            if(fake_rollout_steps == n):
                # Reset fake env to groundtruth
                fake_obs = fake_env.reset(inp = (
                    np.array(real_dynamics_obs_history),
                    np.array(real_dynamics_action_history),
                    real_dynamics_pose,
                    real_info["waypoints"]
                    )
                )

                fake_rollout_steps = 0

        if(generate_videos):
            generate_video(logger, image_path = image_save_dir, save_path = video_save_dir, name = "{}_rollout_{}".format(exp_name, i))

    rollout_mse = n_step_rollout_squared_errors/total_steps
    reward_mse = n_step_reward_squared_errors/total_steps

    rollout_mse = n_step_rollout_squared_errors

    for i in range(n):
        logger.log_scalar(exp_name + "_rollout_mse", rollout_mse[i], i)
        logger.log_scalar(exp_name + "_reward_mse", reward_mse[i], i)


def visualize_trajectory_distribution(exp_name, logger, real_env, fake_env, policy, num_episodes, n = 1, n_samples = 50):
    """Generates images plotting multiple trajectories sampled from the dynamics model against ground truth
    """
    # List to store rollouts errors
    # Store each step separately
    n_step_rollout_squared_errors = np.zeros(shape = (n,))
    n_step_reward_squared_errors = np.zeros(shape = (n,))
    n_step_rollout_squared_errors = n_step_rollout_squared_errors
    n_step_rollout_uncertainty = []
    obs_dim = real_env.observation_space.shape[0]

    # Get the number of stacked frames fake_env needs
    # frame_stack = fake_env.frame_stack
    frame_stack = 10

    image_save_dir = logger.prep_dir(os.path.join("dynamics_eval/images", exp_name))

    total_steps = 0
    real_done = False
    for i in range(num_episodes):

        real_rollout_steps = 0
        img_idx = 0
        print("Experiment: {}, Episode {}".format(exp_name, i))
        real_dynamics_obs_history = deque([], maxlen = frame_stack)
        real_dynamics_action_history = deque([], maxlen = frame_stack)
        real_dynamics_pose = None

        # Reset environment and get real observation
        real_obs = real_env.reset()
        # Do this step because we need to data returned in info, which reset does not return
        real_obs, real_reward, real_done, real_info = real_env.step(np.array([0,0]))


        # Since the fake environment requires a frame stack input, we have to stack up the frames
        # Loop over frame stack to collect the correct rollout stack to start using the fake_env
        for frame in range(frame_stack):
            real_action = policy(real_obs)
            real_dynamics_action_history.append(real_action)

            real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))

            real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])

            real_obs, reward, done, real_info = real_env.step(real_action)

        real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))

        real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])


        real_done = False
        real_rollout_steps = 0
        fake_rollout_steps = 0

        while not real_done and real_rollout_steps < 1000:


            # Initialize variables to store history of rollout
            real_rollout_actions = []
            real_rollout_obs = [real_dynamics_obs_history[-1]]
            real_rollout_pose = [real_dynamics_pose]
            init_real_dynamics_obs_history = copy.deepcopy(real_dynamics_obs_history)
            init_real_action_history = copy.deepcopy(real_dynamics_action_history)
            init_dynamics_pose = copy.deepcopy(real_dynamics_pose)

            # Generate a sequence of actions according to the real environment to evaluate the fake env with
            for rollout_step in range(n):
                action = policy(real_obs)

                real_next_obs, real_reward, real_done, real_info = real_env.step(action)


                next_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                                                    real_info["ego_vehicle_y"],
                                                    real_info["ego_vehicle_theta"]])

                real_obs = real_next_obs
                real_dynamics_pose = next_dynamics_pose
                real_dynamics_obs_history.append(np.array([real_info["steer_angle"], real_info["speed"]]))
                real_dynamics_action_history.append(action)

                real_rollout_actions.append(action)
                real_rollout_obs.append(real_dynamics_obs_history[-1])
                real_rollout_pose.append(real_dynamics_pose)

                real_rollout_steps += 1

                # If we have reached desired fake env rollout length
                if(real_done):
                    break

            # Initialize variables to store history of rollouts
            fake_rollouts_obs = []
            fake_rollouts_pose = []
            for traj_sample in range(n_samples):
                # Initial variables for each rollout
                fake_rollout_obs = [real_rollout_obs[0]]
                fake_rollout_pose = [real_rollout_pose[0]]

                # Reset fake environment with initial history
                fake_obs = fake_env.reset(inp = (
                        np.array(init_real_dynamics_obs_history),
                        np.array(init_real_action_history),
                        init_dynamics_pose,
                        real_info["waypoints"]
                    )
                )

                for step in range(n):
                    fake_next_obs, fake_reward, fake_done, fake_info =  fake_env.step(real_rollout_actions[step])

                    fake_rollout_pose.append(fake_env.vehicle_pose.cpu().numpy())
                    fake_rollout_obs.append(fake_next_obs)

                fake_rollouts_obs.append(fake_rollout_obs)
                fake_rollouts_pose.append(fake_rollout_pose)


            print(fake_rollout_pose)
            real_rollout_pose = np.array(real_rollout_pose)
            fake_rollouts_pose = np.array(fake_rollouts_pose)

            plt.figure()

            xmin = np.amin(real_rollout_pose[:,0])
            xmax = np.amax(real_rollout_pose[:,0])
            ymin = np.amin(real_rollout_pose[:,1])
            ymax = np.amax(real_rollout_pose[:,1])
            for sample in range(fake_rollouts_pose.shape[0]):
                # Plot next state
                x, y, theta = fake_rollouts_pose[sample, :, 0], fake_rollouts_pose[sample, :, 1], fake_rollouts_pose[sample, :, 2]

                temp_min = np.amin(fake_rollouts_pose[sample, :, 0])
                xmin = np.minimum(temp_min, xmin)
                temp_min = np.amax(fake_rollouts_pose[sample, :, 0])
                xmax = np.maximum(temp_min, xmax)

                temp_min = np.amin(fake_rollouts_pose[sample, :, 1])
                ymin = np.minimum(temp_min, ymin)
                temp_min = np.amax(fake_rollouts_pose[sample, :, 1])
                ymax = np.maximum(temp_min, ymax)


                plt.plot(x, y, '-')

            # Plot the true trajectory
            x, y, theta = real_rollout_pose[:,0], real_rollout_pose[:,1], real_rollout_pose[:,2]
            plt.plot(x, y, '-', color = "blue", label = "GT")
            plt.legend()


            # Center agent on the plot
            plt.xlim(xmin-1, xmax+1)
            plt.ylim(ymin-1, ymax+1)

            plt.savefig(os.path.join(image_save_dir, "{:04d}.png".format(img_idx)))
            plt.close()

            img_idx += 1



class DynamicsEvaluationConf:
    def __init__(self):
        self.model_name = "final"
        self.experiment_key = "8f7f242e37434b80ac2109575c3c8942"


def main(args):
    # First, set up comet logger to retrieve experiment
    dynamics_evaluation_conf = DynamicsEvaluationConf()

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = dynamics_evaluation_conf.experiment_key

    logger = CometLogger(logger_conf)

    ### Create the real environment
    env_config = DefaultMainConfig()
    env_config.populate_config(
        observation_config = "VehicleDynamicsNoCameraConfig",
        action_config = "MergedSpeedTanhConfig",
        reward_config = "Simple2RewardConfig",
        testing = False,
        carla_gpu = args.gpu
    )

    env = CarlaEnv(config = env_config, log_dir = logger.log_dir)

    ### Create the fake environment
    # dynamics = DynamicsEnsemble.load(logger, dynamics_evaluation_conf.model_name, gpu = args.gpu)

    dynamics = ProbabilisticGRUDynamicsEnsemble.load(logger, dynamics_evaluation_conf.model_name, gpu = args.gpu)
    # dynamics = GRUDynamicsEnsemble.load(logger, dynamics_evaluation_conf.model_name, gpu = args.gpu)
    # class TempDataModuleConfig():
    #     def __init__(self):
    #         self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/carla-rl_datasets/no_crash_empty"]
    #         self.batch_size = 1
    #         self.frame_stack = 2
    #         self.num_workers = 2
    #         self.train_val_split = 0.95

    # # data config
    # data_config = TempDataModuleConfig()
    # data_module = OfflineCarlaDataModule(data_config)

    # import ipdb; ipdb.set_trace()




    # env setup (obs, action, reward)
    #TODO Fix conflict between fake env Default and Real Env Default
    #TODO Populate with the correct obs_config, action_config, and reward_config
    fake_env_config = DefaultFakeEnvConfig()
    fake_env_config.populate_config(
        observation_config = "VehicleDynamicsNoCameraConfig",
        action_config = "MergedSpeedTanhConfig",
        reward_config="Simple2RewardConfig"
    )

    fake_env = RNNFakeEnv(dynamics,
                config=fake_env_config,
                logger = logger)

    policy = AutopilotPolicy(env)

    visualize_trajectory_distribution("25_step", logger, env, fake_env, policy, 1, n = 25, n_samples = 50)

    # Run desired experiments
    # n_step_eval("TEST_2_autopilot_5_step", logger, env, fake_env, policy, 5, n = 5, generate_videos = True)

    # n_step_eval("TEST_2_autopilot_25_step", logger, env, fake_env, policy, 5, n = 25, generate_videos = True)

    policy = RandomPolicy(env)

    # n_step_eval("V3_random_1_step", logger, env, fake_env, policy, 5, n = 5, generate_videos = True)

    # n_step_eval("V3_random_5_step", logger, env, fake_env, policy, 5, n = 25, generate_videos = True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--n_samples', type=int, default=100000)
    # parser.add_argument('--behavior', type=str, default='cautious')
    # parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)

