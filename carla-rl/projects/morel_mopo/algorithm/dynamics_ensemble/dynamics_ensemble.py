import pytorch_lightning as pl

from logging_utils.comet_checkpoint import CometModelCheckpoint
from dynamics_ensemble import DynamicsEnsembleModule
from data_modules import OfflineCarlaDataModule
import hydra

class DynamicsEnsemble():

    def __init__(self,
                    obs_dim,
                    action_dim,

                    n_models,
                    usad_threshold,

                    epochs,
                    lr,

                    network_cfg,

                    load_path = None,
                    device = "cuda:2"):



        # Dimensions of observations and actions
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.n_models = n_models
        self.usad_threshold = usad_threshold

        self.epochs = epochs

        self.lr = lr

        self.network_cfg = network_cfg

        if(load_path is not None):
            self.dynamics_module = DynamicsEnsembleModule.load_from_checkpoint(load_path)
            self.dynamics_module.to(device)
            self.gpu_number = 2#,
                    # lr = self.lr,
                    # n_models = 15,
                    # usad_threshold = 0.3,
                    # network_cfg = None)
            # class DataModuleConfig:
            #     def __init__(self):
            #         self.dataset_paths = ["/zfsauton/datasets/ArgoRL/swapnilp/pid_no_integral"]
            #         self.use_images =  False
            #         self.batch_size =  256
            #         self.frame_stack =  1
            #         self.num_workers =  5
            #         self.train_val_split =  1.0
            # self.data_module = OfflineCarlaDataModule(DataModuleConfig())
            # self.data_module.setup(None)

        else:
            self.dynamics_module = DynamicsEnsembleModule(lr = self.lr,
                                            n_models = self.n_models,
                                            usad_threshold = self.usad_threshold,
                                            network_cfg = self.network_cfg)


    def train(self, data_module_cfg, logger, gpu_number = None, precision = 16):
        self.gpu_number = gpu_number

        if(precision != 16 and precision != 32):
            raise Exception("Precision must be 16 or 32")

        # self.data_module = hydra.utils.instantiate(data_module_cfg)
        self.data_module = OfflineCarlaDataModule(data_module_cfg)
        self.data_module.setup(None)

        logger.log_hyperparams({
            "lr (dyn_ens)" : self.lr,
            "n_models (dyn_ens)" : self.n_models,
            "batch_size (dyn_ens)" : data_module_cfg.batch_size,
            "train_val_split (dyn_ens)" : data_module_cfg.train_val_split,
            "epochs (dyn_ens)" : self.epochs,
        })

        if(gpu_number is not None):
            gpu_number = [gpu_number]

        model_checkpoint = CometModelCheckpoint(
            "models/dynamics",
            filename = "{epoch}",
            save_top_k = 0,
        )

        trainer = pl.Trainer(
            logger = logger,
            gpus = gpu_number,
            precision = precision,
            # callbacks=callbacks,
            checkpoint_callback = model_checkpoint,
            max_epochs=self.epochs*self.n_models)

        trainer.fit(self.dynamics_module, self.data_module)

    def predict(self, obs):
        self.dynamics_module.eval()
        return self.dynamics_module.predict(obs)

    def get_input_output_dim(self):
        return self.dynamics_module.get_input_output_dim()

    def get_gpu(self):
        return self.gpu_number

    def get_data_module(self):
        return self.data_module

    def get_n_models(self):
        return self.dynamics_module.n_models

    def to(self, device):
        self.dynamics_module.to(device)









