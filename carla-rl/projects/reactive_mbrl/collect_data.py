
import hydra
import click

from projects.reactive_mbrl.data.data_collector import DataCollector
from projects.reactive_mbrl.create_env import create_env


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config):
    output_path = config.data['train_dataset']
    n_samples = int(config.data['n_samples'])
    speed = float(config.data['speed'])
    env = create_env(config.env, output_path)
    collector = DataCollector(env, output_path)
    try:
        total_samples = 0
        while total_samples < n_samples:
            traj_length = collector.collect_trajectory(speed)
            total_samples += traj_length
    except:
        env.close()
        raise
    finally:
        env.close()
        print('Done')


if __name__ == '__main__':
    main()
