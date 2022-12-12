import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import subprocess
import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
import importlib
from environment.env import CarlaEnv
from typing import Optional
from stable_baselines3.common.env_util import DummyVecEnv
from environment.config.scenario_configs import NoCrashEmptyTown01Config,NoCrashDenseTown01Config
from environment.config.observation_configs import LeaderboardObsConfig,VehicleDynamicsObstacleConfig


# # Setup imports for algorithm and environment
# sys.path.append(os.path.abspath(os.path.join('../../../')))

import carla
from common.loggers.comet_logger import CometLogger
from projects.morel_mopo.config.logger_config import CometLoggerConfig,ExistingCometLoggerConfig
from projects.morel_mopo.scripts.dynamics_evaluation import eval_policy_obs



mopo_configs = importlib.import_module("projects.morel_mopo.config.morel_mopo_config")

class AutopilotPolicy:
    def __init__(self, env):
        self.env = env

    def __call__(self, obs):
        return self.env.get_autopilot_action()

    def predict(self, obs, deterministic=True):
        return self.env.get_autopilot_action(),None


def generate_video(logger, image_path, save_path, name):
    vid_path = os.path.join(save_path, name + '.mp4')

    im_path = os.path.join(image_path, "%04d.png")
    gen_vid_command = ["ffmpeg", "-y", "-i", im_path , "-framerate", '20', "-pix_fmt", "yuv420p",
    vid_path]
    gen_vid_process = subprocess.Popen(gen_vid_command, preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
    gen_vid_process.wait()


    logger.log_asset("policy_eval/videos", vid_path)

    # Clear the temporary directory of images after video is generated
    shutil.rmtree(image_path)
    os.mkdir(image_path)


def eval_policy_obs(exp_name, logger, fake_env, policies, num_episodes=1, n = 500,prefix='exp'):

    if(not isinstance(policies, list)):
        policies = [policies]

    image_save_dir = logger.prep_dir(os.path.join("policy_obs_eval/images", exp_name))
    video_save_dir = logger.prep_dir(os.path.join("policy_obs_eval/videos", exp_name))
    image_save_dir = os.path.join(image_save_dir,prefix)
    os.makedirs(image_save_dir, exist_ok=True)

    vid_num = 0
    if True:

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

        #execute = (input() == "y")
        execute = True
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
                    action = policy.predict(np.array([obs]),deterministic=True)
                    actions.append(action[0])

                    

                    obs, reward, done, info = fake_env.step(action[0])

                    print('fake env position',fake_env.vehicle_pose)

                    # Save new values
                    angle_history.append(obs[0])
                    dist_to_trajec_history.append(obs[3])
                    pose_history.append(fake_env.vehicle_pose.cpu().numpy())
                    rewards.append(reward + rewards[-1])
                    uncertains.append(info["uncertain"])
                    obstacle_dist.append(obs[4])
                    obstacle_vel.append(obs[5])

                    if(done):
                        print(info['termination_state'])
                        break

                obs_data.append((angle_history, dist_to_trajec_history, pose_history, actions, rewards, uncertains, obstacle_dist, obstacle_vel))

            if True:
                # Plot obs dist
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][6][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(0, 1.2)
                ax.set_title("Obstacle Dist")
                ax.legend()
                # canvas.draw()
                # obs_dist_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(obs_dist_plot.shape[1] * 0.5)
                # height = int(obs_dist_plot.shape[0] * 0.5)
                # dim = (width, height)
                # obs_dist_plot = cv2.resize(obs_dist_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'obs_distance' + str(i) + '.png'))
                plt.close(fig)

                # Plot obs vel
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][7][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(0, 1.2)
                ax.set_title("Obstacle Velocity")
                ax.legend()
                # canvas.draw()
                # obs_vel_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(obs_vel_plot.shape[1] * 0.5)
                # height = int(obs_vel_plot.shape[0] * 0.5)
                # dim = (width, height)
                # obs_vel_plot = cv2.resize(obs_vel_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'obs_velocity' + str(i) + '.png'))
                plt.close(fig)


                # Plot angle error
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][0][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(-np.pi, np.pi)
                ax.set_title("Angle Error")
                ax.legend()
                # canvas.draw()
                # angle_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(angle_plot.shape[1] * 0.5)
                # height = int(angle_plot.shape[0] * 0.5)
                # dim = (width, height)
                # angle_plot = cv2.resize(angle_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'angle_error' + str(i) + '.png'))
                plt.close(fig)

                # Plot trajectory error
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][1][:i+1], label = str(j))
                ax.set_xlim(0, n)
                ax.set_ylim(-2,2)
                ax.set_title("Trajectory Error")
                ax.legend()
                # canvas.draw()
                # dist_to_trajec_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(dist_to_trajec_plot.shape[1] * 0.5)
                # height = int(dist_to_trajec_plot.shape[0] * 0.5)
                # dim = (width, height)
                # dist_to_trajec_plot = cv2.resize(dist_to_trajec_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'trajectory_error' + str(i) + '.png'))
                plt.close(fig)

                # Plot vehicle position
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
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
                # canvas.draw()
                # wp_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(wp_plot.shape[1] * 0.5)
                # height = int(wp_plot.shape[0] * 0.5)
                # dim = (width, height)
                # wp_plot = cv2.resize(wp_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'vehicle_position' + str(i) + '.png'))
                plt.close(fig)

                # Plot action
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][3][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_ylim(-1,1)
                ax.set_title("Action")
                ax.legend()
                #canvas.draw()
                # action_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(action_plot.shape[1] * 0.5)
                # height = int(action_plot.shape[0] * 0.5)
                # dim = (width, height)
                # action_plot = cv2.resize(action_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'action' + str(i) + '.png'))
                plt.close(fig)

                # Plot cumulative rewards
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][4][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_title("Cumulative Reward")
                ax.legend()
                # canvas.draw()
                # reward_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(reward_plot.shape[1] * 0.5)
                # height = int(reward_plot.shape[0] * 0.5)
                # dim = (width, height)
                # reward_plot = cv2.resize(reward_plot,dim)
                fig.savefig(os.path.join(image_save_dir,'rewards' + str(i) + '.png'))
                plt.close(fig)

                # Plot uncertainty
                fig = plt.figure()
                #canvas = FigureCanvas(fig)
                ax = fig.subplots()
                for j in range(len(policies)):
                    ax.plot(obs_data[j][5][:i+1], label = str(j))
                ax.set_xlim(0,n)
                ax.set_title("Uncertainty")
                ax.legend()
                # canvas.draw()
                # uncertain_plot = np.array(canvas.renderer.buffer_rgba())
                # width = int(uncertain_plot.shape[1] * 0.5)
                # height = int(uncertain_plot.shape[0] * 0.5)
                # dim = (width, height)
                # uncertain_plot = cv2.resize(uncertain_plot,dim)
                plt.close(fig)

                # wp_plot = cv2.vconcat([wp_plot, action_plot])
                # reward_plots = cv2.vconcat([reward_plot, uncertain_plot])
                # obstacle_plots = cv2.vconcat([obs_dist_plot, obs_vel_plot])
                # img = cv2.vconcat([angle_plot, dist_to_trajec_plot])
                # img = cv2.hconcat([wp_plot, img, reward_plots, obstacle_plots])

                # cv2.imwrite(os.path.join(image_save_dir, f"{i:04d}.png"), img)

            #generate_video(logger, image_save_dir, video_save_dir, str(vid_num), "")
            vid_num += 1
            shutil.move(image_save_dir, '/zfsauton2/home/ishaans/output/eval_obs') 



def generate_rollouts(logger, env, policy, n_rollouts = 15, timeout = 10000, prefix='prefix'):
    image_save_dir = logger.prep_dir("policy_eval/images")
    video_save_dir = logger.prep_dir("policy_eval/videos")

    global_idx = 0
    success_eps = 0
    for rollout in tqdm(range(n_rollouts)):
        print(f"Rollout #{rollout}")
        # import ipdb; ipdb.set_trace()
        obs = env.reset()
        
        done = False
        i = 0
        avg_actions,count = np.array([0.0,0.0]),0.0
        while not done:
        # for i in range(timeout):
            action,_ = policy.predict(obs,deterministic=True)

            obs, reward, done, info = env.step(action)
            avg_actions += action
            #print('action',action[0])
            count += 1
            #print(info)
            image = info["sensor.camera.rgb/top"]
            

            im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(image_save_dir, "{:04d}.png".format(i)), im_rgb)
            save_dir = os.path.join(image_save_dir, "{:04d}.png".format(i))
            # print(f"Image to")

            i += 1
            global_idx += 1
            #print(f"Step #{global_idx}")

            if done:
                #print(info[0]["termination_state"])
                #if(info[0]["termination_state"] == "Success"):
                if(info["termination_state"] == "Success"):
                    success_eps += 1
                break
        # term_state = info["termination_state"]
        print('action distribution',avg_actions/count)

    print(f"Success rate: {success_eps/n_rollouts}")

    generate_video(logger = logger,
                    image_path = image_save_dir,
                    save_path = video_save_dir,
                    name = f"{rollout}.mp4")
    shutil.move(video_save_dir,'/zfsauton2/home/ishaans/output/videos/' + prefix)


def main(args):

    config = getattr(mopo_configs, args.variant)()
    config.populate_config(gpu = args.gpu,
                           policy_algorithm = "PPO",
                           pretrained_dynamics_model_key = args.pretrained_key,
                           pretrained_dynamics_model_name = "final")
    config.fake_env_config.uncertainty_coeff = 0

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = args.experiment_key

    logger = CometLogger(logger_conf)


    fake_env_config = config.fake_env_config

    logger_conf = ExistingCometLoggerConfig()
    logger_conf.experiment_key = config.pretrained_dynamics_model_config.key
    temp_logger = CometLogger(logger_conf)

    

    # Load dynamics config
    temp_config = temp_logger.pickle_load("mopo", "config.pkl")
    dynamics_config = temp_config.dynamics_config

    print(f"MOPO: Loading dynamics model {config.pretrained_dynamics_model_config.name} from experiment {config.pretrained_dynamics_model_config.key}")

    dynamics = dynamics_config.dynamics_model_type.load(
                            logger = temp_logger,
                            model_name = config.pretrained_dynamics_model_config.name,
                            gpu = config.gpu)

    policy_data_module_config = config.policy_training_dataset

    fake_env = dynamics_config.fake_env_type(dynamics,
                        config = fake_env_config,
                        policy_data_module_config = policy_data_module_config,
                        logger = logger)

    #autopilot_policy = AutopilotPolicy(fake_env)
    fake_env=DummyVecEnv([lambda: fake_env ])

    policy = config.policy_algorithm.load(logger.other_load("policy/models",'policy_checkpoint__6000000_steps.zip' ),device = f"cuda:{args.gpu}")

    config.eval_env_config.render_server = True
    config.eval_env_config.carla_gpu = args.gpu
    config.eval_env_config.scenario_config = NoCrashDenseTown01Config()
    config.eval_env_config.obs_config = VehicleDynamicsObstacleConfig() 
    config.eval_env_config.scenario_config.set_parameter("disable_traffic_light", True)
    config.eval_env_config.scenario_config.set_parameter("disable_static", True)
    config.eval_env_config.obs_config.set_parameter("disable_lane_invasion_sensor", True)
    env = CarlaEnv(config = config.eval_env_config, log_dir = logger.log_dir)
    eval_env = env.get_eval_env(25)
    generate_rollouts(logger, eval_env, policy,prefix=args.prefix)
    #eval_policy_obs('eval_obs',logger,fake_env,[policy],prefix=args.prefix)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument("--pretrained_key", type=str, required = False,default='8deb43f07842457ab315d6309812831c')
    parser.add_argument("--experiment_key", type=str, required = False,default='8e43da18fc3145d99f2bfb5734f79e16')
    parser.add_argument("--variant", type=str,default='DefaultMLPObstaclesMOPOConfig')
    parser.add_argument("--prefix", type=str,default='dense_env_obs_policy_dataset_v2')
    args = parser.parse_known_args()[0]
    main(args)