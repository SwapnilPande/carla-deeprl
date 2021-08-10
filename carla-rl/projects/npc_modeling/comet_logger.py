  
from pytorch_lightning.loggers import CometLogger


def get_logger(experiment_key=None):
    return CometLogger(
        api_key="7xaxSbfv83yaMIs2XeGFEEoQt",
        project_name="carla-npc",
        workspace="jerrickhoang",
        experiment_key=experiment_key,
    )