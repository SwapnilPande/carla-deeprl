import numpy as np
from tqdm import tqdm

from data_modules import VQVAEDataset, TransformerDataset

PATHS = [
    '/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_expert_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_random/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town01_noisy_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_expert_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_noisy_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town03_random/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_expert_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_noisy_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town04_random/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_expert_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=0.5/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=0.75/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_noisy_speed=1.0/',
    '/zfsauton/datasets/ArgoRL/brianyan/town06_random/'
]

for path in tqdm(PATHS):
    vae_dataset = VQVAEDataset(path)
    np.save('{}/states'.format(path), np.array(vae_dataset.obs).reshape(-1,8))
    np.save('{}/actions'.format(path), np.array(vae_dataset.actions).reshape(-1,2))

    transformer_dataset = TransformerDataset(path)
    np.save('{}/stacked_states'.format(path), np.array(transformer_dataset.obs).reshape(-1,8))
    np.save('{}/stacked_actions'.format(path), np.array(transformer_dataset.actions).reshape(-1,2))

print('Done')
