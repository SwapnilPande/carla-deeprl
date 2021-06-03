# Environment Config
The configuration for the environment is defined as a set of config files located in `carla-rl/environment/config/`. The config is designed to be safe to ensure that users don't:

* Use default parameters unintentionally
* Partially configure portions of the environment that use multiple parameters
* Forget to set parameters, which may cause silent errors in the environment

At the same time, however, it is also designed to be flexible, allowing users to easily modify existing configurations or create new configurations.

## Config Architecture
The config is designed to be a nested group of config objects, that are all subclasses of `BaseConfig`. The config contains many different types of configs, each configuring a different portion of the environment. Each of these config types define a base type, that defines the minimum set of parameters that need to be set for the config of that type. Note that the base type should NOT contain default values for the parameters, instead they should all be instantiated with the value of `None`. The base class for each config type should also contain comments describing the effect of each parameter. For each type of config, the base type should be subclassed to define values of the parameters. The subclasses can additionally define parameters not included in the base config if necessary. The types of configs include `main_configs`, `obs_configs`, `action_configs`, `reward_configs`, and `scenario_configs`.

### BaseConfig
This is the default config object that all config objects are a subclass of. This config defines the following methods:

* `set(name, value)`: Dynamically set a new parameter named `name` with value `value` in the config
* `get(name)`: Retrieve the value of a parameter `name` from the config. This is equivalent to running `config.name`.
* `verify()` : Verifies that no parameter in the config has a value of `None`. This can be overrided by subclasses of `BaseConfig` to do more detailed config verification.

### MainConfig
This is the top-level config, which contains parameters for the server configuration, as well as parameters for the other types of configs. We define a `DefaultMainConfig` that contains the default server parameters, which likely do not need to be modified.

`MainConfig` also defines a `populate` function, to define values for the parameters that are never set by default:

* `observation_config`
* `action_config`
* `reward_config`
* `scenario_config`
* `testing`
* `carla_gpu`

Each of the 4 config types can either be passed a string, containing the class name of the config you would like to select. Alternatively, you can pass a Config object, that is a subclass of the respective base config type, allowing you to modify each config as necessary or implement your own config.

### Observation Config
Contains the parameters that describe the observation space and the sensors attached to the vehicle.

### Action Config
Contains the parameters that describe the action space for the policy.

### Reward Config
Contains the parmaeters that define the reward function for the policy.

### Scenario Config
Contains the parameters that define the scenarios to be run by the environment.


## Usage
Here is an example of instantiating a config and using it in your own training script:


```
# Import Environment and DefaultMainConfig
from environment.env import CarlaEnv
from environment.config.config import DefaultMainConfig

# Instantiate the DefaultMainConfig
config = DefaultMainConfig()
# Populate the needed fields
config.populate_config(
    observation_config = "LowDimObservationConfig", # Use the 8-dim observation space
    action_config = "MergedSpeedScaledTanhConfig", # Use the tanh speed steer space
    reward_config = "Simple2RewardConfig", # Simple2 Rewards
    scenario_config = "NoCrashEmptyTown01Config", # Run NoCrashEmpty Scenarios
    testing = False, # Training, not testing
    carla_gpu = 0 # Run Carla on GPU 0
)

env = CarlaEnv(config = config, log_dir = "/home/foo/bar/carla-rl-logs")
```