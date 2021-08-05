import comet_ml
import torch
import cv2
import hydra
import numpy as np
import os
from omegaconf import OmegaConf

from projects.reactive_mbrl.agents.camera_agent import CameraAgent
from projects.reactive_mbrl.data.comet_logger import get_logger
from projects.reactive_mbrl.create_env import create_env

WIDE_CROP_TOP = 48
NARROW_CROP_BOTTOM = 80


def load_model(model_path):
    output_dir = os.path.dirname(model_path)
    config_path = os.path.join(output_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    model = CameraAgent(config)
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    return model


@hydra.main(config_path="configs", config_name="config.yaml")
def evaluate_model(cfg):
    model_path = cfg.eval.model_path
    output_dir = os.path.dirname(model_path)
    env = create_env(cfg.env, output_dir)
    try:
        evaluate(env, cfg)
    except:
        env.close()
        raise
    finally:
        env.close()
        print("Done")


def evaluate(env, cfg):
    num_scenarios = int(cfg.eval["num_scenarios"])
    model_path = cfg.eval.model_path
    experiment_key = model_path.split("/")[-2]
    comet_logger = get_logger(experiment_key)
    output_dir = os.path.dirname(model_path)

    agent = load_model(model_path)

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
        total_reward = 0.0

        for _ in range(5):
            # Warm start.
            action = env.get_autopilot_action(0.5)
            next_obs, reward, done, info = env.step(action)

        for _ in range(10000):
            wide_rgb, narr_rgb = get_inputs(info)
            action = agent.predict(wide_rgb, narr_rgb)
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
        video_path = os.path.join(
            output_dir, f"scenario_{index}_{cfg.env.family}{cfg.env.town}.avi"
        )
        save_video(frames, video_path)
        comet_logger.experiment.log_asset(video_path, overwrite=True)
    comet_logger.experiment.log_curve(
        x=range(num_scenarios),
        y=reward_list,
        name=f"rewards_{cfg.env.family}{cfg.env.town}",
    )
    comet_logger.experiment.log_metrics(
        termination_counts, prefix=f"{cfg.env.family}{cfg.env.town}",
    )


def get_inputs(info):
    wide_rgb_cropped = np.copy(
        info["sensor.camera.rgb/front_wide"][WIDE_CROP_TOP:, :, ::-1]
    )
    wide_rgb = torch.tensor(wide_rgb_cropped)
    narr_rgb_cropped = np.copy(
        info["sensor.camera.rgb/front_narrow"][:-NARROW_CROP_BOTTOM, :, ::-1]
    )
    narr_rgb = torch.tensor(narr_rgb_cropped)

    wide_rgb = wide_rgb.float().permute(2, 0, 1)
    narr_rgb = narr_rgb.float().permute(2, 0, 1)
    return wide_rgb, narr_rgb


def save_video(frames, fname, fps=15):
    frames = [np.array(frame) for frame in frames]
    height, width = frames[0].shape[0], frames[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    evaluate_model()
