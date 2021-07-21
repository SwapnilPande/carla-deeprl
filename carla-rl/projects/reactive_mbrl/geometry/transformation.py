import numpy as np

def transform_points(transform, points):
    matrix = np.array(transform.get_matrix())
    points_global_T = matrix.dot(points.T)
    return points_global_T.T

def points_to_homogeneous(pixel_xy):
    s, _ = pixel_xy.shape
    return np.concatenate([pixel_xy, np.zeros((s, 1)), np.ones((s, 1))], axis=1)
