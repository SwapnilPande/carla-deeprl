#!/bin/bash


# If -h is passed, list arguments and exit
if [ "$1" == "-h" ]; then
    echo "Usage: run_experiment_batch.sh <experiment_name> <n_experiments> <list_of_gpus>"
    exit 0
fi

# Split arguments into separate variables
# Experiment name
exp_name=$1

# Number of experiments
num_exp=$2

# GPUS to use
gpus=$3

# If GPU contains a comma, split it into an array
IFS=',' read -ra gpu_array <<< "$gpus"

# Confirm that the number of experiments equals the number of GPUs
if [ "$num_exp" -ne "${#gpu_array[@]}" ]; then
    echo "Number of experiments must equal the number of GPUs"
    exit 1
fi
# Construct exp_group_id
# $exp_name_$num_exp_$date
exp_group_id=${exp_name}_${num_exp}_$(date +%Y%m%d)

# Run train_mopo.py with the given parameters n times and log to comet
# Loop num_exp times
for i in $(seq 1 $num_exp); do
    # Run train_mopo.py in a tmux session
    tmux new-session -s ${exp_group_id}_${i} -d "source ~/anaconda3/etc/profile.d/conda.sh && conda activate carla-rl && source ../../../../configure_env.setup && python train_mopo.py --exp_name $exp_name --gpu ${gpu_array[i - 1]} --exp_group $exp_group_id"
    echo "Launched tmux session ${exp_group_id}_${i}"
    sleep 60
done