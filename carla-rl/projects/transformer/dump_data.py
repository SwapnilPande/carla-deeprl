import glob

import numpy as np
from tqdm import tqdm

from data_modules import VQVAEDataset, TransformerDataset
from models.trajectory_transformer import TrajectoryTransformer

PATHS = glob.glob('/zfsauton/datasets/ArgoRL/brianyan/carla_dataset/town01/*/')

# agent = TrajectoryTransformer()

for path in tqdm(PATHS):
    vae_dataset = VQVAEDataset(path)
    np.save('{}/states'.format(path), np.array(vae_dataset.obs).reshape(-1,8))
    np.save('{}/actions'.format(path), np.array(vae_dataset.actions).reshape(-1,2))

    transformer_dataset = TransformerDataset(path)
    np.save('{}/stacked_states'.format(path), np.array(transformer_dataset.obs).reshape(-1,8))
    np.save('{}/stacked_actions'.format(path), np.array(transformer_dataset.actions).reshape(-1,2))

    # stacked_states = np.load('{}/stacked_states.npy'.format(path))
    # stacked_actions = np.load('{}/stacked_actions.npy'.format(path))

    


print('Done')
