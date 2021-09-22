import comet_ml

import hydra
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch

from create_env import create_env
from projects.reactive_mbrl.ego_model import EgoModel
from projects.reactive_mbrl.data.comet_logger import get_logger
from projects.reactive_mbrl.agents.greedy_reward_agent import GreedyRewardAgent
from projects.reactive_mbrl.agents.greedy_dp_agent import GreedyDPAgent
from projects.reactive_mbrl.agents.pid_agent import PIDAgent
from projects.reactive_mbrl.npc_modeling.kinematic import Kinematic


def load_ego_model():
    project_home = os.environ["PROJECT_HOME"]
    model_weights = torch.load(
        os.path.join(project_home, "carla-rl/projects/reactive_mbrl/ego_model.th")
    )
    # model = EgoModel(dt=10.0)
    model = EgoModel(dt=1.0)
    model.load_state_dict(model_weights)
    return model



def evaluate(env, cfg):
    num_scenarios = int(cfg.eval["num_scenarios"])
    comet_logger = get_logger()
    use_pid = bool(cfg.eval["use_pid"])
    # output_dir = cfg.eval["greedy_dir"]
    output_dir = cfg.eval["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reward_list = []
    status_list = []
    termination_counts = {
        "success": 0,
        "obs_collision": 0,
        "lane_invasion": 0,
        "out_of_road": 0,
        "offlane": 0,
        "unexpected_collision": 0,
        "runover_light": 0,
        "max_steps": 0,
        "max_steps_obstacle": 0,
        "max_steps_light": 0,
        "static": 0,
        "unknown": 0,
    }

    for index in range(num_scenarios):
        print(f"Evaluating scenario {index}")
        frames = []
        obs = env.reset(unseen=False, index=index)

        waypoints = env.carla_interface.global_planner._waypoints_queue
        waypoints = np.array(
            [
                [
                    w[0].transform.location.x,
                    w[0].transform.location.y,
                    w[0].transform.rotation.yaw,
                ]
                for w in waypoints
            ]
        )
        model = load_ego_model()
        npc_predictor = Kinematic(model, waypoints)
        if use_pid:
            agent = PIDAgent(npc_predictor, output_dir)
        else:
            agent = GreedyDPAgent(npc_predictor, output_dir)
        scenario_name = f"scenario_{index}_{cfg.environment.family}{cfg.environment.town}"
        agent.reset(waypoints)
        total_reward = 0.0

        print("Warming up")
        for _ in range(1):
            action = env.get_autopilot_action(5)
            obs, reward, done, info = env.step(action)

        print("Done warming up")

        for idx in tqdm(range(10000)):
            speed = info["speed"]
            # action = agent.predict(env, speed, 8)
            action = agent.predict(env, info, speed, 8)
            # action = env.get_autopilot_action(1.5)
            obs, reward, done, info = env.step(action)

            total_reward += reward

            frame = env.render(camera="sensor.camera.rgb/top")
            frames.append(frame)

            if done:
                break

        reward_list.append(total_reward)
        status_list.append(info["termination_state"])
        termination_counts[info["termination_state"]] += 1

        print(
            f"Finished scenario {index}, reward is {reward_list[-1]}, status is {status_list[-1]}"
        )
        video_path = os.path.join(output_dir, f"{scenario_name}.avi")
        save_video(frames, video_path)
        comet_logger.experiment.log_asset(video_path, overwrite=True)
    comet_logger.experiment.log_curve(
        x=range(num_scenarios),
        y=reward_list,
        name=f"rewards_{cfg.environment.family}{cfg.environment.town}",
    )
    comet_logger.experiment.log_metrics(
        termination_counts, prefix=f"{cfg.environment.family}{cfg.environment.town}",
    )


def save_video(frames, fname, fps=15):
    frames = [np.array(frame) for frame in frames]
    height, width = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


@hydra.main(config_path="configs", config_name="no_crash_dense.yaml")
# @hydra.main(config_path="configs", config_name="config.yaml")
def evaluate_agent(cfg):
    output_dir = os.path.dirname(cfg.eval["greedy_dir"])
    env = create_env(cfg.environment, output_dir)
    try:
        evaluate(env, cfg)
    except:
        env.close()
        raise
    finally:
        env.close()
        print("Done")


if __name__ == "__main__":
    evaluate_agent()
