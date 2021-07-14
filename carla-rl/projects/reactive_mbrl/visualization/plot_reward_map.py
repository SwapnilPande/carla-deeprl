import click
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

SEM_COLORS = {
    4: (220, 20, 60),
    5: (153, 153, 153),
    6: (157, 234, 50),
    7: (128, 64, 128),
    8: (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

YAWS = np.linspace(-1.0, 1.0, 5)
SPEEDS = np.linspace(0, 8, 4)

num_yaws = len(YAWS)
num_spds = len(SPEEDS)


def convert_sem(sem, labels=[4, 6, 7, 10, 18]):
    canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
    for i, label in enumerate(labels):
        canvas[sem == i + 1] = SEM_COLORS[label]

    return canvas


@click.command()
@click.option("--dataset-path", type=str, required=True)
@click.option("--output-path", type=str, required=True)
def plot_reward_map(dataset_path, output_path):
    trajectory_paths = glob.glob(f"{dataset_path}/*")
    trajectory_path = trajectory_paths[0]
    idx = 0

    narr_rgb_path = sorted(glob.glob(f"{trajectory_path}/narrow_rgb/*.png"))[idx]
    narr_rgb = cv2.imread(narr_rgb_path)

    narr_seg_path = sorted(glob.glob(f"{trajectory_path}/narrow_seg/*.png"))[idx]
    narr_seg = cv2.imread(narr_seg_path, cv2.IMREAD_GRAYSCALE)
    narr_seg = convert_sem(narr_seg)

    wide_rgb_path = sorted(glob.glob(f"{trajectory_path}/wide_rgb/*.png"))[idx]
    wide_rgb = cv2.imread(wide_rgb_path)

    wide_seg_path = sorted(glob.glob(f"{trajectory_path}/wide_seg/*.png"))[idx]
    wide_seg = cv2.imread(wide_seg_path, cv2.IMREAD_GRAYSCALE)
    wide_seg = convert_sem(wide_seg)

    top_rgb_path = sorted(glob.glob(f"{trajectory_path}/topdown/*.png"))[idx]
    top_rgb = cv2.imread(top_rgb_path)

    value_path = sorted(glob.glob(f"{trajectory_path}/value/*.npy"))[idx]
    value = np.load(value_path)
    action_value_path = sorted(glob.glob(f"{trajectory_path}/action_value/*.npy"))[idx]
    action_value = np.load(action_value_path)
    action_value = action_value[8, 8, 3, 2]
    action_value = action_value[:27].reshape(9, 3)

    plot_value_map(
        narr_rgb,
        wide_rgb,
        narr_seg,
        wide_seg,
        top_rgb,
        value,
        action_value,
        output_path,
    )


def plot_value_map(
    narr_rgb, wide_rgb, narr_seg, wide_seg, top_rgb, value, action_value, output_path
):
    fig = plt.figure(constrained_layout=True, figsize=((25, 10)))
    gs = fig.add_gridspec(nrows=4, ncols=17)

    for s, speed, in enumerate(SPEEDS):
        for y, yaw in enumerate(YAWS):
            m = value[:, :, s, y]
            ax = fig.add_subplot(gs[s, 2 * y : 2 * (y + 1)])
            ax.pcolormesh(m)
            ax.set_title(f"speed={speed:2.2f}, yaw={yaw:2.2f}", fontsize=7)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    ax = fig.add_subplot(gs[0, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(wide_rgb)

    ax = fig.add_subplot(gs[1, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(wide_seg)

    ax = fig.add_subplot(gs[2, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(narr_rgb)

    ax = fig.add_subplot(gs[3, 10:15])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(narr_seg)

    ax = fig.add_subplot(gs[0:2, 15:])
    im = ax.pcolormesh(action_value)
    fig.colorbar(im, ax=ax)
    ax.set_yticklabels([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(gs[2:, 15:])
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.imshow(top_rgb)

    # fig.tight_layout()
    img_path = os.path.join(f"{output_path}/value_map.png")
    plt.savefig(img_path)
    plt.clf()


if __name__ == "__main__":
    plot_reward_map()
