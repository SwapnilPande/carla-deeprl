"""
This file contains a wrapper for stable-baselines3 to automatically restart training from checkpoints periodically.
This exists because of a memory leak present in CARLA 9.10, described here: https://github.com/carla-simulator/leaderboard/issues/81
Our tempororary fix is to periodically restart the server and client.
"""

import multiprocessing

def train_restart_wrapper(train_fn, total_timesteps, restart_interval):
    model_load_name = None
    for i in range(total_timesteps // restart_interval):
        print(f"Starting training interation for steps {i * restart_interval} to {(i+1) * restart_interval}")

        model_save_name = f"policy_{(i + 1) * restart_interval}_steps.zip"
        # Train for restart_interval steps
        process = multiprocessing.Process(target=train_fn, args=(restart_interval, model_save_name, model_load_name))
        process.start()
        process.join()


        model_load_name = model_save_name
