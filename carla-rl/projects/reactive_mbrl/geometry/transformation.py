import numpy as np

def transform_points(transform, points):
    matrix = np.array(transform.get_matrix())
    points_global_T = matrix.dot(points.T)
    return points_global_T.T
