import numpy as np


SEM_COLORS = {
    4 : (220, 20, 60),
    5 : (153, 153, 153),
    6 : (157, 234, 50),
    7 : (128, 64, 128),
    8 : (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

def visualize_semantic(sem, labels=[4,6,7,10,18]):
    canvas = np.zeros(sem.shape + (3,), dtype=np.uint8)
    for label in labels:
        canvas[sem==label] = SEM_COLORS[label]

    return canvas

def visualize_semantic_processed(sem, labels=[4,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = SEM_COLORS[label]

    return canvas
