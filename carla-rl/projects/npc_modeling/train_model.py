import comet_ml
from projects.npc_modeling.comet_logger import get_logger

from omegaconf import OmegaConf
import torch
import hydra
import os

from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from projects.npc_modeling.models.simple_model import SimpleModel
from projects.npc_modeling.data.dataset import NPCDatasetModule


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg):
    dataset = NPCDatasetModule(
        # cfg.data["train_dataset"], val_path=cfg.data["val_dataset"]
        cfg.data["train_dataset"]
    )
    comet_logger = get_logger()
    agent = SimpleModel(cfg, comet_logger)

    output_dir = setup_output_dir(
        cfg.data["model_dir"], comet_logger.experiment.get_key()
    )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(period=1, save_top_k=-1)
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=comet_logger,
        callbacks=callbacks,
        max_epochs=int(cfg.train["num_epochs"]),
        val_check_interval=0.25,
        gpus=[0],
    )
    trainer.fit(agent, dataset)

    save_model(agent, output_dir, comet_logger)
    save_config(cfg, output_dir, comet_logger)


def save_config(config, output_dir, comet_logger):
    config_path = os.path.join(output_dir, "config.yaml")
    OmegaConf.save(config=config, f=config_path)
    comet_logger.experiment.log_asset(config_path)


def save_model(agent, output_dir, comet_logger):
    model_path = os.path.join(output_dir, "model.th")
    torch.save(agent.state_dict(), model_path)

    comet_logger.experiment.log_model(f"model.th", model_path)


def setup_output_dir(model_dir, experiment_key):
    output_dir = os.path.join(model_dir, experiment_key)
    assert not os.path.exists(output_dir)
    os.makedirs(output_dir)
    return output_dir


if __name__ == "__main__":
    main()
