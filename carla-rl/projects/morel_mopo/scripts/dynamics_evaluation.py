import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas
import subprocess
import shutil
import argparse
from collections import deque
import copy
import cv2

# Setup imports for algorithm and environment
sys.path.append(os.path.abspath(os.path.join('../../../')))

from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import ExistingCometLoggerConfig
from projects.morel_mopo.config.dynamics_config import DefaultMLPDynamicsConfig, DefaultDeterministicGRUDynamicsConfig, DefaultProbabilisticMLPDynamicsConfig
from projects.morel_mopo.config.data_module_config import ObstaclesMixedDeterministicMLPDataModuleConfig
from projects.morel_mopo.config.fake_env_config import DefaultFakeEnvConfig
from projects.morel_mopo.algorithm.data_modules import OfflineCarlaDataModule
from projects.morel_mopo.algorithm.mopo import MOPO


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


class MOPOPolicy:
    def __init__(self, logger, mopo_evaluation_conf):
        self.mopo = MOPO.load(logger = logger,
                    policy_model_name = mopo_evaluation_conf.policy_model_name,
                    gpu = args.gpu,
                    policy_only = mopo_evaluation_conf.policy_only)

        self.mopo.config.eval_env_config.carla_gpu = args.gpu

    def __call__(self, obs):
        return self.mopo.policy_predict(np.expand_dims(obs, axis = 0))

################################################################################

def generate_video(logger, image_path, save_path, name, path):
        vid_path = os.path.join(save_path, name + '.mp4')

        im_path = os.path.join(image_path, "%04d.png")
        gen_vid_command = ["ffmpeg", "-y", "-i", im_path , "-framerate", '10', "-pix_fmt", "yuv420p",
        vid_path]
        gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        gen_vid_process.wait()


        # logger.log_asset("dynamics_eval/videos", vid_path)

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
    frame_stack = fake_env.frame_stack

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
            real_dynamics_action_history.appendleft(real_action)

            real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

            real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                             real_info["ego_vehicle_y"],
                             real_info["ego_vehicle_theta"]])

            real_obs, reward, done, real_info = real_env.step(real_action)

        real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

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

        while not real_done and real_rollout_steps < 1000:

            action = policy(real_obs)


            print(real_obs)
            print(fake_obs)
            input()

            # Generate new observations
            real_next_obs, real_reward, real_done, real_info =  real_env.step(action)
            fake_next_obs, fake_reward, fake_done, fake_info =  fake_env.step(action)

            print(action)



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
            real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))
            real_dynamics_action_history.appendleft(action)



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
    frame_stack = fake_env.frame_stack

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
            real_dynamics_action_history.appendleft(real_action)

            real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

            real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])

            real_obs, reward, done, real_info = real_env.step(real_action)

        real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))

        real_dynamics_pose = np.array([real_info["ego_vehicle_x"],
                            real_info["ego_vehicle_y"],
                            real_info["ego_vehicle_theta"]])

        print(np.array(real_dynamics_obs_history))
        real_done = False
        real_rollout_steps = 0
        fake_rollout_steps = 0

        while not real_done and real_rollout_steps < 1000:

            # Initialize variables to store history of rollout
            real_rollout_actions = []
            real_rollout_obs = [real_dynamics_obs_history[0]]
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

                real_dynamics_obs_history.appendleft(np.array([real_info["steer_angle"], real_info["speed"]]))
                real_dynamics_action_history.appendleft(action)

                real_rollout_actions.append(action)
                real_rollout_obs.append(real_dynamics_obs_history[0])
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

def eval_policy_obs(exp_name, logger, fake_env, policies, num_episodes, n = 50):

    if(not isinstance(policies, list)):
        policies = [policies]

    image_save_dir = logger.prep_dir(os.path.join("policy_obs_eval/images", exp_name))
    video_save_dir = logger.prep_dir(os.path.join("policy_obs_eval/videos", exp_name))

    vid_num = 0
    while(True):

        # Plot upcoming waypoints to see if we want to visualize this trajectory
        obs = fake_env.reset()

        wps = fake_env.waypoints[:10].cpu().numpy()
        wp_x = wps[:,0]
        wp_y = wps[:,1]

        npc_poses = fake_env.npc_poses[:n].cpu().numpy()

        plt.figure()
        plt.scatter(wp_x, wp_y)

        if(npc_poses.shape[1] != 0):
            plt.scatter(npc_poses[0,:,0], npc_poses[0,:,1], color = 'green')

        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()

        delta_x = x_max - x_min
        if(delta_x < 4):
            extra = (4 - delta_x)/2
            x_min -= extra
            x_max += extra

        delta_y = y_max - y_min
        if(delta_y < 4):
            extra = (4 - delta_y)/2
            y_min -= extra
            y_max += extra

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.savefig(os.path.join(image_save_dir, "upcoming_waypoints.png"))

        plt.close()

        print(f"Vehicle Pose: {fake_env.vehicle_pose}")

        execute = (input() == "y")
        # We do want to visualize
        if(execute):
            inp = fake_env.get_current_state()
            obs_data = []
            for policy in policies:
                fake_env.reset(inp = inp)

                # Loop for steps
                # Initialize arrays to store data
                angle_history = [obs[0]]
                dist_to_trajec_history = [obs[3]]
                pose_history = [fake_env.vehicle_pose.cpu().numpy()]
                obstacle_dist = [obs[4]]
                obstacle_vel = [obs[5]]
                actions = []
                rewards = [0]
                uncertains = [0]

                for i in range(n):
                    action = policy(obs)
                    actions.append(action[0])

                    obs, reward, done, info = fake_env.step(action)

                    # Save new values
                    angle_history.append(obs[0])
                    dist_to_trajec_history.append(obs[3])
                    pose_history.append(fake_env.vehicle_pose.cpu().numpy())
                    rewards.append(reward + rewards[-1])
                    uncertains.append(info["uncertain"])
                    obstacle_dist.append(obs[4])
                    obstacle_vel.append(obs[5])

                    if(done):
                        break

                obs_data.append((angle_history, dist_to_trajec_history, pose_history, actions, rewards, uncertains, obstacle_dist, obstacle_vel))


            for i in range(n):
                # Plot obs dist
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][6][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(0, 1.2)
                ax.set_title("Obstacle Dist")
                ax.legend()
                canvas.draw()
                obs_dist_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(obs_dist_plot.shape[1] * 0.5)
                height = int(obs_dist_plot.shape[0] * 0.5)
                dim = (width, height)
                obs_dist_plot = cv2.resize(obs_dist_plot,dim)
                plt.close(fig)

                # Plot obs vel
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][7][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(0, 1.2)
                ax.set_title("Obstacle Velocity")
                ax.legend()
                canvas.draw()
                obs_vel_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(obs_vel_plot.shape[1] * 0.5)
                height = int(obs_vel_plot.shape[0] * 0.5)
                dim = (width, height)
                obs_vel_plot = cv2.resize(obs_vel_plot,dim)
                plt.close(fig)


                # Plot angle error
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][0][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(-np.pi, np.pi)
                ax.set_title("Angle Error")
                ax.legend()
                canvas.draw()
                angle_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(angle_plot.shape[1] * 0.5)
                height = int(angle_plot.shape[0] * 0.5)
                dim = (width, height)
                angle_plot = cv2.resize(angle_plot,dim)
                plt.close(fig)

                # Plot trajectory error
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][1][:i+1], label = str(j))
                ax.set_xlim(0, n)
                ax.set_ylim(-2,2)
                ax.set_title("Trajectory Error")
                ax.legend()
                canvas.draw()
                dist_to_trajec_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(dist_to_trajec_plot.shape[1] * 0.5)
                height = int(dist_to_trajec_plot.shape[0] * 0.5)
                dim = (width, height)
                dist_to_trajec_plot = cv2.resize(dist_to_trajec_plot,dim)
                plt.close(fig)

                # Plot vehicle position
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                ax.scatter(wp_x, wp_y, color = 'red')
                for j in range(len(policies)):
                    if(len(obs_data[j][2]) > i):
                        ax.arrow(obs_data[j][2][i][0],
                                obs_data[j][2][i][1],
                                np.cos(np.deg2rad(obs_data[j][2][i][2])),
                                np.sin(np.deg2rad(obs_data[j][2][i][2])),
                                 label = str(j))
                    else:
                        ax.arrow(obs_data[j][2][-1][0],
                                obs_data[j][2][-1][1],
                                np.cos(np.deg2rad(obs_data[j][2][-1][2])),
                                np.sin(np.deg2rad(obs_data[j][2][-1][2])),
                                label = str(j))

                if(len(obs_data[0][2]) > i):
                    ax.set_xlim(obs_data[0][2][i][0] - 10, obs_data[0][2][i][0] + 10)
                    ax.set_ylim(obs_data[0][2][i][1] - 10 , obs_data[0][2][i][1] + 10)
                else:
                    ax.set_xlim(obs_data[0][2][-1][0] - 10, obs_data[0][2][-1][0] + 10)
                    ax.set_ylim(obs_data[0][2][-1][1] - 10 , obs_data[0][2][-1][1] + 10)

                if(npc_poses.shape[1] > 0):
                    ax.scatter(npc_poses[i, :, 0], npc_poses[i, :, 1], color = 'green')
                ax.set_title("Vehicle position")
                ax.legend()
                canvas.draw()
                wp_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(wp_plot.shape[1] * 0.5)
                height = int(wp_plot.shape[0] * 0.5)
                dim = (width, height)
                wp_plot = cv2.resize(wp_plot,dim)
                plt.close(fig)

                # Plot action
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][3][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(-1,1)
                ax.set_title("Action")
                ax.legend()
                canvas.draw()
                action_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(action_plot.shape[1] * 0.5)
                height = int(action_plot.shape[0] * 0.5)
                dim = (width, height)
                action_plot = cv2.resize(action_plot,dim)
                plt.close(fig)

                # Plot cumulative rewards
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][4][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_title("Cumulative Reward")
                ax.legend()
                canvas.draw()
                reward_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(reward_plot.shape[1] * 0.5)
                height = int(reward_plot.shape[0] * 0.5)
                dim = (width, height)
                reward_plot = cv2.resize(reward_plot,dim)
                plt.close(fig)

                # Plot uncertainty
                fig = plt.figure()
                canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][5][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_title("Uncertainty")
                ax.legend()
                canvas.draw()
                uncertain_plot = np.array(canvas.renderer.buffer_rgba())
                width = int(uncertain_plot.shape[1] * 0.5)
                height = int(uncertain_plot.shape[0] * 0.5)
                dim = (width, height)
                uncertain_plot = cv2.resize(uncertain_plot,dim)
                plt.close(fig)

                wp_plot = cv2.vconcat([wp_plot, action_plot])
                reward_plots = cv2.vconcat([reward_plot, uncertain_plot])
                obstacle_plots = cv2.vconcat([obs_dist_plot, obs_vel_plot])
                img = cv2.vconcat([angle_plot, dist_to_trajec_plot])
                img = cv2.hconcat([wp_plot, img, reward_plots, obstacle_plots])

                cv2.imwrite(os.path.join(image_save_dir, f"{i:04d}.png"), img)

            generate_video(logger, image_save_dir, video_save_dir, str(vid_num), "")
            vid_num += 1

class DynamicsEvaluationConf:
    def __init__(self):
        self.model_name = "final"
        self.experiment_key = "cb248318a263406795aaabc4e8a51a3f"

class MOPOEvaluationConf:
    def __init__(self):
        self.policy_model_name = "policy_checkpoint__7000000_steps.zip"
        self.experiment_key = "cb248318a263406795aaabc4e8a51a3f"
        self.policy_only = True


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




    # env = CarlaEnv(config = env_config, log_dir = logger.log_dir)


    dynamics_config = DefaultMLPDynamicsConfig()
    data_module_config = ObstaclesMixedDeterministicMLPDataModuleConfig()
    # data_module_config = None
    ### Create the fake environment
    dynamics = dynamics_config.dynamics_model_type.load(logger, dynamics_evaluation_conf.model_name, gpu = args.gpu, data_config = data_module_config)



    dynamics_config.dynamics_model_config = dynamics.config



    # env setup (obs, action, reward)
    #TODO Fix conflict between fake env Default and Real Env Default
    #TODO Populate with the correct obs_config, action_config, and reward_config
    fake_env_config = DefaultFakeEnvConfig()
    fake_env_config.populate_config(
        observation_config = "VehicleDynamicsObstacleNoCameraConfig",
        action_config = "MergedSpeedTanhConfig",
        reward_config="Simple2RewardConfig"
    )

    fake_env = dynamics_config.fake_env_type(dynamics,
                config=fake_env_config,
                logger = logger)

    mopo_evaluation_conf = MOPOEvaluationConf()
    policy = MOPOPolicy(logger, mopo_evaluation_conf)
    autopilot_policy = AutopilotPolicy(fake_env)

    eval_policy_obs("test", logger, fake_env, [policy, autopilot_policy], 10, 50)
    exit()

    # Run desired experiments
    # n_step_eval("V3_autopilot_5_step", logger, env, fake_env, policy, 5, n = 5, generate_videos = True)

    # n_step_eval("V3_autopilot_25_step", logger, env, fake_env, policy, 5, n = 25, generate_videos = True)

    # visualize_trajectory_distribution("25_step", logger, env, fake_env, policy, 1, n = 25, n_samples = 50)

    # policy = RandomPolicy(env)

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

