# loggers
The loggers are used to log data from your experiments. Currently, the only supported logger is the `CometLogger`, which logs data to [comet.ml](https://www.comet.ml/). You can sign up for a free student account on comet to get started.

## Architecture
This logger directory contains a `BaseLogger`, from which all other loggers inherit. The `BaseLogger` only currently has the funtionality of creating a log directory when you start an experiment.

The `CometLogger` inherits from the `BaseLogger` to extend the functionality to write data to comet.ml.

Tensorboard support is possible, although not currently implemented.

## Comet Logger
The comet logger logs data to comet.ml. The logger supports logging the following objects:
* Hyperparameters
* Scalar Metrics
* Pickle Objects
* Torch Models

Additionally, you can use the comet logger to retrieve the models and pickle objects from comet for your testing/evaluation. This allows you to abstract away file operations by letting the logger handle them, meaning you don't have to keep track of where your models are saved. However, you can still choose to save and load your models and files manually in your code without using the logger.

### Config
The base config can be found in the `comet_logger_config.py` file. To set up a comet logger, you should create your own logger config that inherits from the `BaseCometLoggerConfig` and defines all of the values. You can find an example comet_logger_config below. We don't include a default config because the api_key is different for each user. When creating a new experiment, you must pass the api_key, workspace, project_name, and log_dir.

```
from common.loggers.comet_logger_config import BaseCometLoggerConfig

class CometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "<INSERT API KEY HERE>"
        self.workspace = "<INSERT COMET WORKSPACE HERE>"
        self.project_name = "<INSERT COMET PROJECT NAME HERE>"
        self.log_dir = "/home/scratch/<USERNAME>/carla-rl_logs"

```
### API
#### `__init__`
Constructs a comet logger.

Args:
* config - A comet logger config object.

#### `log_hyperparameters`
Logs hyperparameters to comet.

Args:
* hyperparameters - Dictionary containing the name (key) and the value (value) for each hyperparameter.

Returns
* None

#### `log_scalar`
Logs scalar values to comet

Args:
* name - Name of the scalar being logged
* value - Value of the scalar being logged
* step - Timestep associated with the scalar. If not passed, the parameter will be logged without a timestep.


Returns
* None

#### `torch_save`
Saves the state_dict for a `torch.nn.Module`, both in the local log directory as well as to comet.

Args:
* log_path - Directory (relative to the root of the log directory) where the model should be saved. The same directory is used to log the model in comet
* model_name - Name to save the model under. This will be the comet model name as well as the file name

Returns:
* None

#### `torch_load`
Loads a torch model from the logs. First, this will attempt to load the model locally if it exists. Else, it will download the model from comet to a temporary directory to load it.

Args:
* log_path - Directory (relative to the root of the log directory) where the model is located.
* model_name - Name of the model

Returns:
* A torch state_dict containing the parameters for the model.

#### `torch_save`
Saves any object as a pickle, both in the local log directory as well as to comet.

Args:
* log_path - Directory (relative to the root of the log directory) where the model should be saved. The same directory is used to log the model in comet
* model_name - Name to save the model under. This will be the comet model name as well as the file name

Returns:
* None

#### `torch_load`
Loads a pickled object from the logs. First, this will attempt to load the pickle locally if it exists. Else, it will download the pickle from comet to a temporary directory to load it.

Args:
* log_path - Directory (relative to the root of the log directory) where the pickle is located.
* model_name - Name of the pickle

Returns:
* The unpickled object

## Opening Existing Experiments
The comet logger also has the functionality of allowing you to load past experiments, potentially to load models or log additional data during evaluation. To do this, you still instantiate a comet logger, but pass in the experiment key in your config as well. An example config for this is below:

```
class ExistingCometLoggerConfig(BaseCometLoggerConfig):
    def __init__(self):
        super().__init__()
        self.api_key = "<INSERT API KEY HERE>"
        self.experiment_key = "<INSERT EXPERIMENT KEY HERE>"
        self.log_dir = "/home/scratch/<USERNAME>/carla-rl_logs"
```

Notice that you pass in the experiment key, and do not pass the workspace or project name.

Once the experiment is loaded, you can use the above API just as the original experiment.



## Doc Revisions
If something seems incorrect on unclear in these docs, contact Swapnil Pande (swapnilp@andrew.cmu.edu).