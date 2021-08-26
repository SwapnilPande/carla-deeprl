from pytorch_lightning.loggers import CometLogger


def get_logger(experiment_key=None):
    return CometLogger(
        api_key="kcCmjGSRjPK0OfY95sDnMtgBY",
        project_name="carla-reactive",
        workspace="bhyang",
        experiment_key=experiment_key,
    )
