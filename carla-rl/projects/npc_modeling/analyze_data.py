import hydra
from glob import glob
import os

from projects.npc_modeling.data.dataset import NPCDataset
import projects.npc_modeling.geometry.geometry as geometry


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config):
    dataset_path = config.data["train_dataset"]
    dataset = NPCDataset(dataset_path)

    item = dataset[0]
    positions = geometry.position2d_from_transform(item)
    normalized = geometry.position2d_from_transform(geometry.normalize_trajectory(item))
    positions_normalized = positions - positions[0]

    import matplotlib
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3)

    ax = axes[0]
    ax.plot(normalized[:, 0], normalized[:, 1], "o")
    ax.axis("equal")

    ax = axes[1]
    ax.plot(positions[:, 0], positions[:, 1], "o")
    ax.axis("equal")

    ax = axes[2]
    ax.plot(positions_normalized[:, 0], positions_normalized[:, 1], "o")
    ax.axis("equal")

    plt.savefig("/zfsauton/datasets/ArgoRL/jhoang/logs/positions.png")


if __name__ == "__main__":
    main()
