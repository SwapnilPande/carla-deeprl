import numpy as np


def position2d_from_transform(transforms):
    return transforms[:, :2, 3]


def normalize_trajectory(transforms):
    first = transforms[0]
    first_inv = np.linalg.inv(first)

    normalized = np.einsum("jk,ikl->ijl", first_inv, transforms)
    return normalized

