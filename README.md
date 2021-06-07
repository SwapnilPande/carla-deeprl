# carla-rl

## Setup Instructions

* Prerequisites: Follow the instructions at https://github.com/Auton-Self-Driving/alta/edit/carla_upgrade/ to install Carla.


* Modify your existing configure_env.setup file to contain the following:
```
export ALTA=$HOME/alta
export LIBS=$ALTA/libs
export PATH=$LIBS/nasm/bin:$LIBS/libjpeg8/bin:$LIBS/libpng/bin:$LIBS/libjpeg/bin:$LIBS/libjpeglua/bin:$PATH
export LD_LIBRARY_PATH=$LIBS/libjpeg8/lib:$LIBS/libpng/lib:$LIBS/libjpeg/lib64:$LIBS/libjpeglua/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=$LIBS/libjpeg8/include:$LIBS/libpng/include:$LIBS/libjpeg/include:$LIBS/libjpeglua/include:$CPLUS_INCLUDE_PATH
export CARLA_9_4_PATH=$HOME/carla910
export SDL_AUDIODRIVER='dsp'
export CARLA_ROOT=$HOME/carla910
export PYTHONPATH=$PYTHONPATH:$HOME/carla910/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
```
*  Execute this file by running
```
source configure_env.setup
```

* Git clone carla-deeprl repository inside $HOME/projects and switch to 'offline_mbrl' branch.
```
mkdir $HOME/projects
cd $HOME/projects
git clone --recurse-submodule https://github.com/Auton-Self-Driving/carla-deeprl.git
cd carla-deeprl
git checkout offline_mbrl
```

* Install conda environment 'carla-rl' from carla-rl.yml file.

```
conda env create -f carla-rl.yml 
conda activate carla-rl
```

## Testing installation 

* Navigate to carla-rl/projects/morel_mopo/scripts, and run: 
```
python train_dynamics.py
```
