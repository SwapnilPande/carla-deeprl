# carla-rl
This repo is designed for researching reinforcement learning techniques for self-driving cars in the [Carla Simulator](https://carla.org/). The repo implements an environment meeting the gym specifications that wraps the carla simulator, standard benchmarks used in the RL for self-driving literature, utilities for logging/monitoring experiments, and the various projects in the lab. The following docs contain information about how to use the environment utilities for your own research as well as information about the design of the codebase in case you find a need to extend it.

## Repository Organization
* Top-level directory contains external packages that are used in this repo as well as system environment configuration scripts/files.
* /carla-rl - contains the project code
    * /algorithms - Contains implementations of RL algorithms to be shared across projects. This contains stable-baselines3 as well, which implements most standard model-free RL algorithms. For the sake of being able to compare across projects, these implementations should be used whenever possible.
    * /common - Contains standard utilities that are shared across projects
    * /environment - Contains all of the code for constructing the gym environment and interacting with the Carla simulator.
    * /projects - Contains the individual projects across the lab.

The objective in this architecture is to share as much code across projects as possible. However, each project can contain any necessary code for itself separately from the rest of the repository, so as not to interfere with other projects. Changes should never be made to non-project specific code without ensuring that the changes will not impact the other lab members.

The repo is designed such that each project can design it's own training loop and interact with the environment using the standard gym interface. The repo also implements a logger across the projects, to which the environment will log data. Each project can log additional data to the loggers.

## Setup Instructions
These instructions are written for the Auton lab cluster, you may have to modify these for different systems.

### CARLA Installation
Currently, this repo supports Carla 9.10. Support for older/future versions is in the pipeline. Installation instructions for different versions of CARLA are very similar.

* Change to home directory ($HOME).

```
cd ~
```

* Install CARLA v0.9.10 (https://carla.org/2020/09/25/release-0.9.10/) for which the binaries are available here: (https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz)

```
mkdir $HOME/carla910
cd $HOME/carla910
wget "https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz"
tar xvzf CARLA_0.9.10.1.tar.gz
```

### Cloning This Repo
* Clone carla-deeprl repository in your home directory.
```
cd ~
git clone --recurse-submodule https://github.com/Auton-Self-Driving/carla-deeprl.git
cd carla-deeprl
```

### Configuring Conda Environment, Installing Dependencies, and setting environment variables
The repo contains a conda environment including all of the dependencies for the project.

First, create the conda environment using the `carla-rl.yml` file:
```
conda env create -f carla-rl.yml
```

Next, we need to install stable-baselines3 in editable mode within the conda environment
```
conda activate carla-rl
cd ~/carla-deeprl/carla-rl/algorithms/stable-baselines3
pip install -e .
```

Finally, we need to configure the environment varialbes for the project. These are contained in `configure_env.setup`. First, verify that all the paths in the file match your project setup. You can add the contents of this file to your `~/.bashrc` or `~/.zshrc` to automatically configure these variables when you launch your shell. Alternatively, you can run the following before running code from the repo:
```
source configure_env.setup
```

### Testing installation
TODO


## Before You Start
You should read the following docs to understand how to configure the environment and the logger.
* [Environment Config Documentation](/carla-rl/environment/config/README.md)

## Logger Docs
* [Logger docs](/carla-rl/common/loggers/README.md)

